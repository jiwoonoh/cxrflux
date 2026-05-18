#!/usr/bin/env python3

import argparse
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoencoder import CXRLatentAutoencoder
from bridge import BridgeProcess
from dataset import CXRPairDataset, build_transform, prepare_pair_splits
from experiment_utils import apply_conditioning_mode, move_batch_to_device
from latent_diffusion import (
    LatentAnchoredPotentialOutcomeMeanPredictor,
    LatentConditionalUNet,
    LatentDDPM,
    LatentMeanPredictor,
    LatentPotentialOutcomeMeanPredictor,
)
from unet import ConditionalUNet


BRIDGE_REFERENCE = {
    "generated_mae": 0.1538,
    "generated_ssim": 0.4436,
    "generated_dark_field_dice": 0.7389,
    "generated_lower_saturation_excess": 0.0030,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate latent DDPM CXR predictions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument(
        "--treatment-column",
        default=None,
        help=(
            "CSV column used as scalar treatment condition. Defaults to the "
            "checkpoint config value, then 'treated'."
        ),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--foreground-crop", action="store_true")
    parser.add_argument("--crop-threshold", type=int, default=None)
    parser.add_argument("--crop-min-content-fraction", type=float, default=None)
    parser.add_argument("--crop-margin-fraction", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--sample-start-timestep", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument(
        "--counterfactual-treatment-value",
        type=float,
        default=None,
        help=(
            "If set, evaluate swapped/counterfactual samples by replacing the "
            "treatment condition with this scalar instead of using 1-a."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--skip-image-verify", action="store_true")
    parser.add_argument(
        "--xrv-semantic-weights",
        default="",
        help="Optional TorchXRayVision DenseNet weights for semantic CXR labeler scoring.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denorm(tensor):
    return tensor.detach().float().clamp(-1.0, 1.0).add(1.0).div(2.0)


def slugify_label(label):
    return re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")


def load_xrv_semantic_bundle(weights, device):
    if not weights:
        return None
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights=weights).to(device)
    model.eval()
    columns = []
    keep_indices = []
    for index, label in enumerate(model.pathologies):
        slug = slugify_label(label)
        if not slug:
            continue
        columns.append(f"xrv_{slug}")
        keep_indices.append(index)
    resolution = int(getattr(model, "input_resolution", 224) or 224)
    return {
        "model": model,
        "columns": columns,
        "keep_indices": keep_indices,
        "resolution": resolution,
        "weights": weights,
    }


@torch.no_grad()
def xrv_semantic_scores(bundle, prefix, image):
    if bundle is None:
        return {}
    x = image.detach().float().clamp(0.0, 1.0)
    if x.shape[-1] != bundle["resolution"] or x.shape[-2] != bundle["resolution"]:
        x = F.interpolate(
            x,
            size=(bundle["resolution"], bundle["resolution"]),
            mode="bilinear",
            align_corners=False,
        )
    x = (2.0 * x - 1.0) * 1024.0
    predictions = bundle["model"](x)[:, bundle["keep_indices"]].float()
    return {
        f"{prefix}_{column}": predictions[:, index]
        for index, column in enumerate(bundle["columns"])
    }


def create_ssim_window(window_size, sigma, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords = coords - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.view(1, 1, window_size, window_size)


def compute_ssim(prediction, target, window):
    c1 = 0.01**2
    c2 = 0.03**2
    padding = window.shape[-1] // 2
    mu_prediction = F.conv2d(prediction, window, padding=padding)
    mu_target = F.conv2d(target, window, padding=padding)
    sigma_prediction_sq = F.conv2d(prediction * prediction, window, padding=padding) - mu_prediction.pow(2)
    sigma_target_sq = F.conv2d(target * target, window, padding=padding) - mu_target.pow(2)
    sigma_prediction_target = F.conv2d(prediction * target, window, padding=padding) - mu_prediction * mu_target
    ssim_map = (
        (2 * mu_prediction * mu_target + c1)
        * (2 * sigma_prediction_target + c2)
        / (
            (mu_prediction.pow(2) + mu_target.pow(2) + c1)
            * (sigma_prediction_sq + sigma_target_sq + c2)
        )
    )
    return ssim_map.mean(dim=(1, 2, 3))


def total_variation(image):
    diff_x = image[..., 1:] - image[..., :-1]
    diff_y = image[:, :, 1:, :] - image[:, :, :-1, :]
    return diff_x.abs().mean(dim=(1, 2, 3)) + diff_y.abs().mean(dim=(1, 2, 3))


def high_frequency_energy(image):
    smooth = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
    return (image - smooth).abs().mean(dim=(1, 2, 3))


def border_mask_like(image, fraction=0.06):
    height, width = image.shape[-2:]
    border = max(1, int(round(fraction * min(height, width))))
    mask = torch.zeros_like(image)
    mask[..., :border, :] = 1.0
    mask[..., -border:, :] = 1.0
    mask[..., :, :border] = 1.0
    mask[..., :, -border:] = 1.0
    return mask


def soft_dark_field_mask(image):
    darker_than_soft_tissue = torch.sigmoid((0.58 - image) / 0.08)
    brighter_than_border = torch.sigmoid((image - 0.08) / 0.03)
    mask = darker_than_soft_tissue * brighter_than_border
    height, width = image.shape[-2:]
    roi = torch.zeros_like(mask)
    roi[..., int(0.06 * height) : int(0.94 * height), int(0.04 * width) : int(0.96 * width)] = 1.0
    return mask * roi


def lower_chest_roi_mask_like(image):
    height, width = image.shape[-2:]
    mask = torch.zeros_like(image)
    y0 = int(round(0.48 * height))
    y1 = int(round(0.86 * height))
    x0 = int(round(0.12 * width))
    x1 = int(round(0.88 * width))
    mask[..., y0:y1, x0:x1] = 1.0
    return mask


def clinical_proxy_tensors(prefix, image):
    roi = lower_chest_roi_mask_like(image)
    roi_denom = roi.sum(dim=(1, 2, 3)).clamp_min(1e-8)
    dark_mask = soft_dark_field_mask(image)
    return {
        f"{prefix}_lower_chest_brightness": (image * roi).sum(dim=(1, 2, 3)) / roi_denom,
        f"{prefix}_lower_dark_fraction": (dark_mask * roi).sum(dim=(1, 2, 3)) / roi_denom,
        f"{prefix}_global_dark_fraction": dark_mask.mean(dim=(1, 2, 3)),
    }


def soft_dice(mask_a, mask_b):
    numerator = 2.0 * (mask_a * mask_b).sum(dim=(1, 2, 3))
    denominator = mask_a.sum(dim=(1, 2, 3)) + mask_b.sum(dim=(1, 2, 3))
    return numerator / denominator.clamp_min(1e-8)


def compute_artifacts(target, generated, target_mask=None, generated_mask=None):
    height = generated.shape[-2]
    lower_start = int(round(0.78 * height))
    lower_target = target[..., lower_start:, :]
    lower_generated = generated[..., lower_start:, :]
    lower_band_abs_diff = (lower_generated - lower_target).abs().mean(dim=(1, 2, 3))
    lower_saturation_excess = (
        (lower_generated > 0.96).float().mean(dim=(1, 2, 3))
        - (lower_target > 0.96).float().mean(dim=(1, 2, 3))
    ).clamp_min(0.0)
    border = border_mask_like(generated)
    border_abs_diff = ((generated - target).abs() * border).sum(dim=(1, 2, 3))
    border_abs_diff = border_abs_diff / border.sum(dim=(1, 2, 3)).clamp_min(1e-8)
    high_frequency_ratio = high_frequency_energy(generated) / high_frequency_energy(target).clamp_min(1e-8)
    if target_mask is None:
        target_mask = soft_dark_field_mask(target)
    if generated_mask is None:
        generated_mask = soft_dark_field_mask(generated)
    dark_area_ratio = generated_mask.sum(dim=(1, 2, 3)) / target_mask.sum(dim=(1, 2, 3)).clamp_min(1e-8)
    return {
        "lower_band_abs_diff": lower_band_abs_diff,
        "border_abs_diff": border_abs_diff,
        "lower_saturation_excess": lower_saturation_excess,
        "high_frequency_ratio": high_frequency_ratio,
        "dark_field_area_ratio": dark_area_ratio,
    }


def _bridge_config_value(config, key, default):
    return config[key] if key in config and config[key] is not None else default


def load_bridge(checkpoint_path, device, method_override="", inference_steps_override=0):
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = ConditionalUNet(
        in_channels=1,
        base_channels=int(_bridge_config_value(config, "base_channels", 64)),
    ).to(device)
    model.load_state_dict(checkpoint.get("ema_model_state_dict") or checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bridge = BridgeProcess(
        num_steps=int(_bridge_config_value(config, "bridge_steps", 100)),
        noise_scale=float(_bridge_config_value(config, "bridge_noise_scale", 0.06)),
        endpoint_probability=float(_bridge_config_value(config, "endpoint_probability", 0.35)),
        residual_scale=float(_bridge_config_value(config, "residual_scale", 1.0)),
        residual_mask_floor=float(_bridge_config_value(config, "residual_mask_floor", 1.0)),
        residual_edge_margin=float(_bridge_config_value(config, "residual_edge_margin", 0.06)),
        residual_lower_taper_start=float(_bridge_config_value(config, "residual_lower_taper_start", 0.70)),
    )
    method = method_override or str(_bridge_config_value(config, "selection_method", "one_step"))
    inference_steps = int(
        inference_steps_override
        if inference_steps_override
        else _bridge_config_value(config, "inference_steps", 25)
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "model": model,
        "bridge": bridge,
        "config": config,
        "method": method,
        "inference_steps": inference_steps,
        "conditioning_mode": str(_bridge_config_value(config, "conditioning_mode", "full")),
    }


def load_latent_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    autoencoder_checkpoint = Path(config["autoencoder_checkpoint"])
    if not autoencoder_checkpoint.exists():
        fallback = Path(__file__).resolve().parents[1] / "checkpoints" / "autoencoder" / "best_model.pt"
        if fallback.exists():
            autoencoder_checkpoint = fallback
    ae_checkpoint = torch.load(autoencoder_checkpoint, map_location=device)
    ae_config = ae_checkpoint.get("config", {})
    autoencoder = CXRLatentAutoencoder(
        base_channels=int(ae_config.get("base_channels", 64)),
        latent_channels=int(ae_config.get("latent_channels", config.get("latent_channels", 4))),
    ).to(device)
    autoencoder.load_state_dict(
        ae_checkpoint.get("ema_model_state_dict") or ae_checkpoint["model_state_dict"]
    )
    autoencoder.eval()
    latent_channels = int(config.get("latent_channels", ae_config.get("latent_channels", 4)))
    model = LatentConditionalUNet(
        latent_channels=latent_channels,
        conditioning_latents=int(config.get("conditioning_latents", 1)),
        base_channels=int(config.get("latent_base_channels", 128)),
    ).to(device)
    model.load_state_dict(checkpoint.get("ema_model_state_dict") or checkpoint["model_state_dict"])
    model.eval()
    model._checkpoint_config = config
    mean_head = None
    if config.get("target_mode", "target") == "mean_residual":
        mean_architecture = config.get("mean_architecture", "standard")
        if mean_architecture == "potential_outcome":
            mean_head = LatentPotentialOutcomeMeanPredictor(
                latent_channels=latent_channels,
                base_channels=int(config.get("mean_base_channels", 64)),
                treatment_effect_scale=float(config.get("treatment_effect_scale", 1.0)),
            ).to(device)
        elif mean_architecture == "anchored_potential_outcome":
            mean_head = LatentAnchoredPotentialOutcomeMeanPredictor(
                latent_channels=latent_channels,
                base_channels=int(config.get("mean_base_channels", 64)),
                treatment_effect_scale=float(config.get("treatment_effect_scale", 1.0)),
            ).to(device)
        else:
            mean_head = LatentMeanPredictor(
                latent_channels=latent_channels,
                base_channels=int(config.get("mean_base_channels", 64)),
            ).to(device)
        mean_state = checkpoint.get("mean_ema_model_state_dict") or checkpoint["mean_model_state_dict"]
        mean_head.load_state_dict(mean_state)
        mean_head.eval()
    ddpm = LatentDDPM(
        num_timesteps=int(config.get("num_timesteps", 1000)),
        beta_schedule=config.get("beta_schedule", "cosine"),
    ).to(device)
    return checkpoint, config, autoencoder, model, ddpm, mean_head


@torch.no_grad()
def encode_pair(autoencoder, batch):
    return autoencoder.encode(batch["x_0"], sample=False), autoencoder.encode(batch["y"], sample=False)


@torch.no_grad()
def predict_bridge_image(bridge_bundle, batch, device, use_amp):
    treatment = apply_conditioning_mode(batch["a"], bridge_bundle["conditioning_mode"])
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        if bridge_bundle["method"] == "one_step":
            prediction = bridge_bundle["bridge"].one_step_predict(
                bridge_bundle["model"],
                batch["x_0"],
                treatment,
                batch["delta"],
            )
        elif bridge_bundle["method"] == "iterative":
            prediction = bridge_bundle["bridge"].iterative_predict(
                bridge_bundle["model"],
                batch["x_0"],
                treatment,
                batch["delta"],
                inference_steps=bridge_bundle["inference_steps"],
            )
        else:
            raise ValueError(f"Unsupported bridge method: {bridge_bundle['method']}")
    return prediction.float().clamp(-1.0, 1.0)


@torch.no_grad()
def build_latent_inputs(autoencoder, batch, bridge_bundle, device, use_amp):
    baseline_latent, _ = encode_pair(autoencoder, batch)
    conditioning_latent = baseline_latent
    start_latent = baseline_latent
    bridge_image = None
    bridge_latent = None
    if bridge_bundle is not None:
        bridge_image = predict_bridge_image(bridge_bundle, batch, device, use_amp)
        bridge_latent = autoencoder.encode(bridge_image, sample=False)
        conditioning_latent = torch.cat([baseline_latent, bridge_latent], dim=1)
        start_latent = bridge_latent
    return conditioning_latent, start_latent, bridge_latent, bridge_image


@torch.no_grad()
def sample_images(
    autoencoder,
    model,
    mean_head,
    ddpm,
    batch,
    start_timestep,
    steps,
    bridge_bundle,
    device,
    use_amp,
    noise=None,
):
    config = getattr(model, "_checkpoint_config", {})
    target_mode = config.get("target_mode", "target")
    residual_scale = float(config.get("residual_scale", 1.0))
    treatment = apply_conditioning_mode(
        batch["a"],
        config.get("treatment_conditioning_mode", "full"),
    )
    if target_mode == "mean_residual":
        if mean_head is None:
            raise ValueError("Mean-residual latent DDPM evaluation requires a mean predictor")
        baseline_latent, _ = encode_pair(autoencoder, batch)
        mean_latent = mean_head(baseline_latent, treatment, batch["delta"])
        conditioning_latent = torch.cat([baseline_latent, mean_latent], dim=1)
        start_latent = torch.zeros_like(mean_latent)
        bridge_latent = None
        bridge_image = None
    else:
        conditioning_latent, start_latent, bridge_latent, bridge_image = build_latent_inputs(
            autoencoder,
            batch,
            bridge_bundle,
            device,
            use_amp,
        )
        if target_mode == "residual":
            if bridge_latent is None:
                raise ValueError("Residual latent DDPM evaluation requires bridge conditioning")
            start_latent = torch.zeros_like(bridge_latent)
        elif target_mode == "delta":
            start_latent = torch.zeros_like(start_latent)
    sampled_latent = ddpm.ddim_sample(
        model,
        conditioning_latent,
        treatment,
        batch["delta"],
        start_timestep=start_timestep,
        steps=steps,
        start_latent=start_latent,
        noise=noise,
    )
    if target_mode == "delta":
        baseline_latent = conditioning_latent[:, : sampled_latent.shape[1]]
        generated_latent = baseline_latent + residual_scale * sampled_latent
    elif target_mode == "residual":
        generated_latent = bridge_latent + residual_scale * sampled_latent
    elif target_mode == "mean_residual":
        generated_latent = mean_latent + residual_scale * sampled_latent
    else:
        generated_latent = sampled_latent
    return autoencoder.decode(generated_latent), bridge_image


def prepare_eval_data(args, checkpoint_config):
    csv_path = args.csv_path or checkpoint_config.get("csv_path")
    image_root = args.image_root or checkpoint_config.get("image_root", "mimic-cxr-jpg")
    split_dir = args.split_dir or checkpoint_config.get("split_dir")
    treatment_column = args.treatment_column or checkpoint_config.get("treatment_column", "treated")
    image_size = args.image_size or int(checkpoint_config.get("image_size", 256))
    foreground_crop = bool(args.foreground_crop or checkpoint_config.get("foreground_crop", False))
    crop_threshold = (
        args.crop_threshold
        if args.crop_threshold is not None
        else int(checkpoint_config.get("crop_threshold", 10))
    )
    crop_min_content_fraction = (
        args.crop_min_content_fraction
        if args.crop_min_content_fraction is not None
        else float(checkpoint_config.get("crop_min_content_fraction", 0.02))
    )
    crop_margin_fraction = (
        args.crop_margin_fraction
        if args.crop_margin_fraction is not None
        else float(checkpoint_config.get("crop_margin_fraction", 0.03))
    )
    split_frames = prepare_pair_splits(
        csv_path=csv_path,
        image_root=image_root,
        split_dir=split_dir,
        seed=int(checkpoint_config.get("seed", 42)),
        verify_readable=not args.skip_image_verify,
    )
    frame = split_frames[args.split].reset_index(drop=True)
    if args.max_examples is not None:
        frame = frame.iloc[: args.max_examples].copy().reset_index(drop=True)
    dataset = CXRPairDataset(
        frame,
        image_root,
        build_transform(
            image_size,
            foreground_crop=foreground_crop,
            crop_threshold=crop_threshold,
            crop_min_content_fraction=crop_min_content_fraction,
            crop_margin_fraction=crop_margin_fraction,
        ),
        treatment_column=treatment_column,
        lung_mask_root=(
            getattr(args, "lung_mask_root", "")
            if getattr(args, "lung_mask_root", "")
            and str(getattr(args, "lung_mask_root", "")).lower() != "none"
            else None
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, frame, image_size


def summarize(frame):
    numeric_cols = [
        "baseline_mae",
        "bridge_mae",
        "generated_mae",
        "baseline_ssim",
        "bridge_ssim",
        "generated_ssim",
        "bridge_dark_field_dice",
        "generated_dark_field_dice",
        "baseline_dark_field_dice",
        "bridge_lower_saturation_excess",
        "generated_lower_saturation_excess",
        "generated_border_abs_diff",
        "generated_high_frequency_ratio",
        "generated_tv_ratio",
        "generated_change_ratio",
        "swap_l1",
        "baseline_lower_chest_brightness",
        "target_lower_chest_brightness",
        "generated_lower_chest_brightness",
        "swapped_lower_chest_brightness",
        "baseline_lower_dark_fraction",
        "target_lower_dark_fraction",
        "generated_lower_dark_fraction",
        "swapped_lower_dark_fraction",
        "baseline_global_dark_fraction",
        "target_global_dark_fraction",
        "generated_global_dark_fraction",
        "swapped_global_dark_fraction",
        "target_lower_brightness_response",
        "generated_lower_brightness_response",
        "swapped_lower_brightness_response",
        "lower_brightness_response_error",
        "target_lower_dark_response",
        "generated_lower_dark_response",
        "swapped_lower_dark_response",
        "lower_dark_response_error",
    ]
    summary = {"count": int(len(frame))}
    for column in numeric_cols:
        summary[column] = float(frame[column].mean()) if len(frame) and column in frame.columns else None
    semantic_cols = [
        column
        for column in frame.columns
        if column.startswith("xrv_treatment_")
        or column.startswith("xrv_swap_abs_")
        or column.startswith("generated_xrv_")
        or column.startswith("swapped_xrv_")
        or column.startswith("target_xrv_")
        or column.startswith("baseline_xrv_")
    ]
    for column in semantic_cols:
        summary[column] = float(frame[column].mean()) if len(frame) else None
    if len(frame):
        summary["mae_improvement_vs_baseline"] = float(
            (frame["baseline_mae"] - frame["generated_mae"]).mean()
        )
        summary["ssim_improvement_vs_baseline"] = float(
            (frame["generated_ssim"] - frame["baseline_ssim"]).mean()
        )
        if "bridge_mae" in frame.columns:
            summary["mae_improvement_vs_bridge"] = float(
                (frame["bridge_mae"] - frame["generated_mae"]).mean()
            )
            summary["ssim_improvement_vs_bridge"] = float(
                (frame["generated_ssim"] - frame["bridge_ssim"]).mean()
            )
    return summary


def promotion_gate(summary):
    if summary["count"] == 0:
        return False
    reference_gate = bool(
        summary["generated_mae"] <= BRIDGE_REFERENCE["generated_mae"]
        and summary["generated_ssim"] >= BRIDGE_REFERENCE["generated_ssim"]
        and summary["generated_dark_field_dice"] >= BRIDGE_REFERENCE["generated_dark_field_dice"]
        and summary["generated_lower_saturation_excess"]
        <= BRIDGE_REFERENCE["generated_lower_saturation_excess"]
        and summary["generated_mae"] <= summary["baseline_mae"]
        and summary["generated_ssim"] >= summary["baseline_ssim"]
    )
    if not reference_gate:
        return False
    if summary.get("bridge_mae") is None:
        return True
    return bool(
        summary["generated_mae"] <= summary["bridge_mae"]
        and summary["generated_dark_field_dice"] >= summary["bridge_dark_field_dice"]
        and summary["generated_lower_saturation_excess"]
        <= max(
            summary["bridge_lower_saturation_excess"],
            BRIDGE_REFERENCE["generated_lower_saturation_excess"],
        )
    )


def subgroup_summary(results):
    outputs = []
    group_specs = []
    if "treated_int" in results.columns:
        group_specs.append(("treated_int", results["treated_int"]))
    if "hours_diff" in results.columns:
        group_specs.append(
            (
                "time_gap_bin",
                pd.cut(
                    results["hours_diff"],
                    bins=[0.0, 12.0, 24.0, 36.0, 48.0],
                    labels=["0-12h", "12-24h", "24-36h", "36-48h"],
                    include_lowest=True,
                ),
            )
        )
    if "gender" in results.columns:
        group_specs.append(("gender", results["gender"].astype(str)))
    if "age_at_t0" in results.columns:
        group_specs.append(
            (
                "age_bin",
                pd.cut(
                    pd.to_numeric(results["age_at_t0"], errors="coerce"),
                    bins=[0, 50, 65, 80, 120],
                    labels=["0-50", "50-65", "65-80", "80+"],
                    include_lowest=True,
                ),
            )
        )
    if "baseline_mean_intensity" in results.columns and len(results) >= 4:
        group_specs.append(
            (
                "baseline_intensity_quartile",
                pd.qcut(
                    results["baseline_mean_intensity"],
                    q=4,
                    labels=["Q1_dark", "Q2", "Q3", "Q4_bright"],
                    duplicates="drop",
                ),
            )
        )
    for group_name, group_values in group_specs:
        temp = results.copy()
        temp[group_name] = group_values
        for value, group in temp.groupby(group_name, dropna=False):
            row = {"group": group_name, "value": str(value)}
            row.update(summarize(group))
            outputs.append(row)
    return pd.DataFrame(outputs)


def save_panel(records, output_path, sample_count, title, generated_label="Latent DDPM"):
    selected = records[:sample_count]
    if not selected:
        return
    include_bridge = any("bridge_image" in record for record in selected)
    rows = 5 if include_bridge else 4
    fig, axes = plt.subplots(rows, len(selected), figsize=(4 * len(selected), 3 * rows))
    if len(selected) == 1:
        axes = axes.reshape(rows, 1)
    for col, record in enumerate(selected):
        images = [
            (record["baseline_image"], "Baseline"),
        ]
        if include_bridge:
            images.append((record.get("bridge_image", record["baseline_image"]), "Bridge"))
        images.extend(
            [
                (record["target_image"], "Ground Truth"),
                (
                    record["generated_image"],
                    "Bridge + residual DDPM" if include_bridge else generated_label,
                ),
                (record["abs_error_image"], "Abs Error"),
            ]
        )
        for row, (image, label) in enumerate(images):
            axes[row, col].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, col].set_title(label if row else f"a={record['treated_int']}, dt={record['hours_diff']:.0f}h")
            axes[row, col].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_treatment_swap_panel(records, output_path, sample_count):
    selected = [record for record in records if "swapped_image" in record][:sample_count]
    if not selected:
        return
    fig, axes = plt.subplots(5, len(selected), figsize=(4 * len(selected), 15))
    if len(selected) == 1:
        axes = axes.reshape(5, 1)
    for col, record in enumerate(selected):
        swap_map = abs(record["swapped_image"] - record["generated_image"])
        images = [
            (record["baseline_image"], "Baseline", "gray", 0.0, 1.0),
            (record["target_image"], "Ground Truth", "gray", 0.0, 1.0),
            (record["generated_image"], f"Factual a={record['treated_int']}", "gray", 0.0, 1.0),
            (record["swapped_image"], f"Swapped a={1 - record['treated_int']}", "gray", 0.0, 1.0),
            (swap_map, "|Swapped - factual|", "magma", 0.0, max(0.05, float(swap_map.max()))),
        ]
        for row, (image, label, cmap, vmin, vmax) in enumerate(images):
            axes[row, col].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, col].set_title(label)
            axes[row, col].axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.disable_amp) and device.type == "cuda"
    checkpoint, config, autoencoder, model, ddpm, mean_head = load_latent_checkpoint(args.checkpoint, device)
    bridge_bundle = load_bridge(
        config.get("bridge_checkpoint", ""),
        device,
        method_override=config.get("bridge_method", "") or config.get("bridge_method_resolved", ""),
        inference_steps_override=int(
            config.get("bridge_inference_steps", 0)
            or config.get("bridge_inference_steps_resolved", 0)
            or 0
        ),
    )
    loader, split_frame, image_size = prepare_eval_data(args, config)
    start_timestep = args.sample_start_timestep or int(config.get("sample_start_timestep", 250))
    sample_steps = args.sample_steps or int(config.get("sample_steps", 50))
    ssim_window = create_ssim_window(11, 1.5, device)
    xrv_bundle = load_xrv_semantic_bundle(args.xrv_semantic_weights, device)

    records = []
    image_records = []
    cursor = 0
    for batch in tqdm(loader, desc=f"evaluate latent DDPM {args.split}"):
        batch = move_batch_to_device(batch, device)
        batch_size = batch["x_0"].shape[0]
        metadata = split_frame.iloc[cursor : cursor + batch_size].reset_index(drop=True)
        cursor += batch_size
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            paired_noise = torch.randn_like(autoencoder.encode(batch["x_0"], sample=False))
            generated_raw, bridge_raw = sample_images(
                autoencoder,
                model,
                mean_head,
                ddpm,
                batch,
                start_timestep,
                sample_steps,
                bridge_bundle,
                device,
                use_amp,
                noise=paired_noise,
            )
            swapped_batch = dict(batch)
            if args.counterfactual_treatment_value is None:
                swapped_batch["a"] = 1.0 - batch["a"]
            else:
                swapped_batch["a"] = torch.full_like(
                    batch["a"],
                    float(args.counterfactual_treatment_value),
                )
            swapped_raw, _ = sample_images(
                autoencoder,
                model,
                mean_head,
                ddpm,
                swapped_batch,
                start_timestep,
                sample_steps,
                bridge_bundle,
                device,
                use_amp,
                noise=paired_noise,
            )

        baseline = denorm(batch["x_0"])
        target = denorm(batch["y"])
        generated = denorm(generated_raw)
        swapped = denorm(swapped_raw)
        bridge_prediction = denorm(bridge_raw) if bridge_raw is not None else None
        baseline_mae = (baseline - target).abs().mean(dim=(1, 2, 3))
        generated_mae = (generated - target).abs().mean(dim=(1, 2, 3))
        baseline_ssim = compute_ssim(baseline, target, ssim_window)
        generated_ssim = compute_ssim(generated, target, ssim_window)
        target_mask = soft_dark_field_mask(target)
        baseline_mask = soft_dark_field_mask(baseline)
        generated_mask = soft_dark_field_mask(generated)
        baseline_dark_dice = soft_dice(baseline_mask, target_mask)
        generated_dark_dice = soft_dice(generated_mask, target_mask)
        artifacts = compute_artifacts(target, generated, target_mask, generated_mask)
        baseline_artifacts = compute_artifacts(target, baseline, target_mask, baseline_mask)
        if bridge_prediction is not None:
            bridge_mae = (bridge_prediction - target).abs().mean(dim=(1, 2, 3))
            bridge_ssim = compute_ssim(bridge_prediction, target, ssim_window)
            bridge_mask = soft_dark_field_mask(bridge_prediction)
            bridge_dark_dice = soft_dice(bridge_mask, target_mask)
            bridge_artifacts = compute_artifacts(target, bridge_prediction, target_mask, bridge_mask)
        else:
            bridge_mae = None
            bridge_ssim = None
            bridge_dark_dice = None
            bridge_artifacts = None
        generated_tv_ratio = total_variation(generated) / total_variation(target).clamp_min(1e-8)
        generated_change = (generated - baseline).abs().mean(dim=(1, 2, 3))
        target_change = (target - baseline).abs().mean(dim=(1, 2, 3))
        generated_change_ratio = generated_change / target_change.clamp_min(1e-8)
        swap_l1 = (swapped - generated).abs().mean(dim=(1, 2, 3))
        clinical_metrics = {}
        for prefix, image in (
            ("baseline", baseline),
            ("target", target),
            ("generated", generated),
            ("swapped", swapped),
        ):
            clinical_metrics.update(clinical_proxy_tensors(prefix, image))
        xrv_metrics = {}
        if xrv_bundle is not None:
            for prefix, image in (
                ("baseline", baseline),
                ("target", target),
                ("generated", generated),
                ("swapped", swapped),
            ):
                xrv_metrics.update(xrv_semantic_scores(xrv_bundle, prefix, image))
            for column in xrv_bundle["columns"]:
                label = column.removeprefix("xrv_")
                generated_score = xrv_metrics[f"generated_{column}"]
                swapped_score = xrv_metrics[f"swapped_{column}"]
                treated_mask = (batch["a"] >= 0.5).view(-1)
                score_y1 = torch.where(treated_mask, generated_score, swapped_score)
                score_y0 = torch.where(treated_mask, swapped_score, generated_score)
                xrv_metrics[f"xrv_treatment_delta_{label}"] = score_y1 - score_y0
                xrv_metrics[f"xrv_treatment_reduction_{label}"] = score_y0 - score_y1
                xrv_metrics[f"xrv_swap_abs_{label}"] = (generated_score - swapped_score).abs()
        target_lower_brightness_response = (
            clinical_metrics["baseline_lower_chest_brightness"]
            - clinical_metrics["target_lower_chest_brightness"]
        )
        generated_lower_brightness_response = (
            clinical_metrics["baseline_lower_chest_brightness"]
            - clinical_metrics["generated_lower_chest_brightness"]
        )
        swapped_lower_brightness_response = (
            clinical_metrics["baseline_lower_chest_brightness"]
            - clinical_metrics["swapped_lower_chest_brightness"]
        )
        target_lower_dark_response = (
            clinical_metrics["target_lower_dark_fraction"]
            - clinical_metrics["baseline_lower_dark_fraction"]
        )
        generated_lower_dark_response = (
            clinical_metrics["generated_lower_dark_fraction"]
            - clinical_metrics["baseline_lower_dark_fraction"]
        )
        swapped_lower_dark_response = (
            clinical_metrics["swapped_lower_dark_fraction"]
            - clinical_metrics["baseline_lower_dark_fraction"]
        )

        for row in range(batch_size):
            meta = metadata.iloc[row].to_dict()
            record = {
                "example_index": int(len(records)),
                "subject_id": str(batch["subject_id"][row]),
                "cxr_0": str(batch["cxr_0"][row]),
                "cxr_1": str(batch["cxr_1"][row]),
                "treated_int": int(batch["a"][row].item() >= 0.5),
                "hours_diff": float(batch["hours_diff"][row].item()),
                "baseline_mae": float(baseline_mae[row].item()),
                "generated_mae": float(generated_mae[row].item()),
                "baseline_ssim": float(baseline_ssim[row].item()),
                "generated_ssim": float(generated_ssim[row].item()),
                "baseline_dark_field_dice": float(baseline_dark_dice[row].item()),
                "generated_dark_field_dice": float(generated_dark_dice[row].item()),
                "generated_lower_band_abs_diff": float(artifacts["lower_band_abs_diff"][row].item()),
                "generated_border_abs_diff": float(artifacts["border_abs_diff"][row].item()),
                "generated_lower_saturation_excess": float(artifacts["lower_saturation_excess"][row].item()),
                "generated_high_frequency_ratio": float(artifacts["high_frequency_ratio"][row].item()),
                "generated_dark_field_area_ratio": float(artifacts["dark_field_area_ratio"][row].item()),
                "baseline_lower_saturation_excess": float(baseline_artifacts["lower_saturation_excess"][row].item()),
                "generated_tv_ratio": float(generated_tv_ratio[row].item()),
                "generated_change_ratio": float(generated_change_ratio[row].item()),
                "swap_l1": float(swap_l1[row].item()),
                "baseline_mean_intensity": float(baseline[row].mean().item()),
                "target_mean_intensity": float(target[row].mean().item()),
                "generated_mean_intensity": float(generated[row].mean().item()),
                "target_lower_brightness_response": float(
                    target_lower_brightness_response[row].item()
                ),
                "generated_lower_brightness_response": float(
                    generated_lower_brightness_response[row].item()
                ),
                "swapped_lower_brightness_response": float(
                    swapped_lower_brightness_response[row].item()
                ),
                "lower_brightness_response_error": float(
                    (
                        generated_lower_brightness_response[row]
                        - target_lower_brightness_response[row]
                    ).item()
                ),
                "target_lower_dark_response": float(target_lower_dark_response[row].item()),
                "generated_lower_dark_response": float(
                    generated_lower_dark_response[row].item()
                ),
                "swapped_lower_dark_response": float(swapped_lower_dark_response[row].item()),
                "lower_dark_response_error": float(
                    (generated_lower_dark_response[row] - target_lower_dark_response[row]).item()
                ),
            }
            for key, value in clinical_metrics.items():
                record[key] = float(value[row].item())
            for key, value in xrv_metrics.items():
                record[key] = float(value[row].item())
            if bridge_prediction is not None:
                record.update(
                    {
                        "bridge_mae": float(bridge_mae[row].item()),
                        "bridge_ssim": float(bridge_ssim[row].item()),
                        "bridge_dark_field_dice": float(bridge_dark_dice[row].item()),
                        "bridge_lower_saturation_excess": float(
                            bridge_artifacts["lower_saturation_excess"][row].item()
                        ),
                    }
                )
            for key in ("gender", "age_at_t0", "view_0", "view_1", "treatment_status", "balancing_weight"):
                if key in meta:
                    record[key] = meta[key]
            image_record = dict(record)
            image_record.update(
                {
                    "baseline_image": baseline[row, 0].detach().cpu().numpy(),
                    "target_image": target[row, 0].detach().cpu().numpy(),
                    "generated_image": generated[row, 0].detach().cpu().numpy(),
                    "swapped_image": swapped[row, 0].detach().cpu().numpy(),
                    "abs_error_image": (generated[row, 0] - target[row, 0]).abs().detach().cpu().numpy(),
                }
            )
            if bridge_prediction is not None:
                image_record["bridge_image"] = bridge_prediction[row, 0].detach().cpu().numpy()
            records.append(record)
            image_records.append(image_record)

    results = pd.DataFrame(records)
    examples_path = output_dir / (
        "strict_test_comparison.csv" if args.split == "test" else "latent_eval_examples.csv"
    )
    results.to_csv(examples_path, index=False)
    summary = summarize(results)
    summary.update(
        {
            "split": args.split,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "bridge_reference": BRIDGE_REFERENCE,
            "passes_promotion_gate": promotion_gate(summary),
            "sample_start_timestep": start_timestep,
            "sample_steps": sample_steps,
            "treatment_conditioning_mode": config.get("treatment_conditioning_mode", "full"),
            "xrv_semantic_weights": args.xrv_semantic_weights or None,
        }
    )
    summary_name = "strict_test_comparison_summary.json" if args.split == "test" else "latent_eval_summary.json"
    (output_dir / summary_name).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "latent_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    subgroup = subgroup_summary(results)
    subgroup.to_csv(output_dir / "subgroup_results.csv", index=False)

    random_records = image_records[: args.sample_count]
    best_records = sorted(image_records, key=lambda item: item["generated_mae"])[: args.sample_count]
    worst_records = sorted(image_records, key=lambda item: item["generated_mae"], reverse=True)[: args.sample_count]
    target_mode = config.get("target_mode", "target")
    if target_mode == "delta":
        generated_label = "Baseline + delta DDPM"
    elif target_mode == "mean_residual":
        if config.get("mean_architecture", "standard") in {
            "potential_outcome",
            "anchored_potential_outcome",
        }:
            generated_label = "Potential mean + residual DDPM"
        else:
            generated_label = "Mean + residual DDPM"
    else:
        generated_label = "Latent DDPM"
    save_panel(
        random_records,
        output_dir / "random_case_panel.png",
        args.sample_count,
        "Random latent DDPM cases",
        generated_label=generated_label,
    )
    save_panel(
        best_records,
        output_dir / "best_case_panel.png",
        args.sample_count,
        "Best latent DDPM cases",
        generated_label=generated_label,
    )
    save_panel(
        worst_records,
        output_dir / "worst_case_panel.png",
        args.sample_count,
        "Worst latent DDPM cases",
        generated_label=generated_label,
    )
    save_treatment_swap_panel(image_records, output_dir / "treatment_swap_panel.png", args.sample_count)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
