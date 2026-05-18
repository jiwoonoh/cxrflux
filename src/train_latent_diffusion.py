#!/usr/bin/env python3

import argparse
import copy
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
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


FINAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = FINAL_ROOT / "results" / "runs" / "latent_diffusion_cxr256_v1"


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a baseline-conditioned latent DDPM.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--image-root", default="mimic-cxr-jpg")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--autoencoder-checkpoint", required=True)
    parser.add_argument(
        "--treatment-column",
        default="treated",
        help=(
            "CSV column used as the scalar treatment condition. The default "
            "binary column is 'treated'; dose-response runs can pass a "
            "continuous normalized dose column."
        ),
    )
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--bridge-checkpoint", default="")
    parser.add_argument("--bridge-method", choices=["one_step", "iterative"], default="")
    parser.add_argument("--bridge-inference-steps", type=int, default=0)
    parser.add_argument(
        "--target-mode",
        choices=["target", "delta", "residual", "mean_residual"],
        default="target",
    )
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--mean-architecture",
        choices=["standard", "potential_outcome", "anchored_potential_outcome"],
        default="standard",
    )
    parser.add_argument(
        "--treatment-conditioning-mode",
        choices=["full", "no_treatment"],
        default="full",
        help="Whether latent mean/diffusion modules receive factual treatment or a zeroed treatment input.",
    )
    parser.add_argument("--mean-base-channels", type=int, default=64)
    parser.add_argument("--mean-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--diffusion-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Weight for the residual DDPM denoising loss. Set to 0 for "
            "mean-head-only causal/treatment-effect fine-tuning."
        ),
    )
    parser.add_argument(
        "--mean-image-l1-weight",
        type=float,
        default=0.0,
        help=(
            "Add decoded image-space L1 supervision to the deterministic mean "
            "prediction in mean_residual mode."
        ),
    )
    parser.add_argument(
        "--mean-highpass-weight",
        type=float,
        default=0.0,
        help=(
            "Add decoded high-frequency L1 supervision to the deterministic mean "
            "prediction in mean_residual mode to discourage blurry mean CXRs."
        ),
    )
    parser.add_argument(
        "--mean-highpass-kernel",
        type=int,
        default=5,
        help="Odd average-pooling kernel used for mean-image high-pass supervision.",
    )
    parser.add_argument("--treatment-effect-l1-weight", type=float, default=0.0)
    parser.add_argument(
        "--treatment-effect-highpass-weight",
        type=float,
        default=0.0,
        help=(
            "Penalize high-frequency decoded-image changes induced by tau, "
            "preserving anatomy while allowing low-frequency edema response."
        ),
    )
    parser.add_argument(
        "--treatment-effect-highpass-kernel",
        type=int,
        default=5,
        help="Odd average-pooling kernel used for tau high-pass consistency.",
    )
    parser.add_argument(
        "--treatment-effect-nonlung-weight",
        type=float,
        default=0.0,
        help=(
            "Penalize decoded treatment-effect magnitude outside a label-free "
            "lung-field proxy. Intended for anchored potential-outcome adapters."
        ),
    )
    parser.add_argument(
        "--treatment-effect-border-weight",
        type=float,
        default=0.0,
        help="Penalize decoded treatment-effect magnitude in image border regions.",
    )
    parser.add_argument(
        "--treatment-effect-mask-method",
        choices=["hybrid", "dark-field", "anatomic", "precomputed"],
        default="hybrid",
        help=(
            "Mask used for lung-localized treatment-effect regularization. "
            "Use precomputed with --lung-mask-root for cached anatomical segmentations."
        ),
    )
    parser.add_argument(
        "--lung-mask-root",
        default="",
        help=(
            "Directory containing cached baseline lung masks keyed by cxr_0 DICOM id. "
            "Required when --treatment-effect-mask-method precomputed is used."
        ),
    )
    parser.add_argument(
        "--treatment-effect-r-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Add an orthogonalized R-learner loss for the latent treatment "
            "effect: ||(Y - mu0) - (A - e(X)) tau||^2."
        ),
    )
    parser.add_argument(
        "--treatment-propensity-column",
        default="p_treated",
        help=(
            "CSV column containing e(X)=P(A=1|X) for binary R-learner runs, "
            "or m(X)=E[D|X] for continuous dose-response R-learner runs."
        ),
    )
    parser.add_argument(
        "--treatment-propensity-eps",
        type=float,
        default=0.02,
        help="Clamp propensity scores to [eps, 1-eps] before residualizing treatment.",
    )
    parser.add_argument("--treatment-effect-scale", type=float, default=1.0)
    parser.add_argument(
        "--treated-loss-multiplier",
        type=float,
        default=1.0,
        help="Multiply train/validation losses for factual treated examples.",
    )
    parser.add_argument(
        "--freeze-potential-progression",
        action="store_true",
        help="For potential-outcome mean heads, freeze mu0 parameters and train only tau.",
    )
    parser.add_argument(
        "--freeze-ddpm",
        action="store_true",
        help="Freeze the latent DDPM residual model during fine-tuning.",
    )
    parser.add_argument("--allow-partial-init", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "ldm_frontal_pairs_v1"))
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--foreground-crop", action="store_true")
    parser.add_argument("--crop-threshold", type=int, default=10)
    parser.add_argument("--crop-min-content-fraction", type=float, default=0.02)
    parser.add_argument("--crop-margin-fraction", type=float, default=0.03)
    parser.add_argument("--latent-base-channels", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=None)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--loss-weight-column", default="")
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--sample-start-timestep", type=int, default=250)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--xrv-semantic-loss-weight",
        type=float,
        default=0.0,
        help="Weight for frozen TorchXRayVision semantic supervision on the potential mean branch.",
    )
    parser.add_argument(
        "--xrv-semantic-weights",
        default="",
        help="TorchXRayVision DenseNet weights, e.g. densenet121-res224-mimic_ch.",
    )
    parser.add_argument(
        "--xrv-semantic-labels",
        default="edema,effusion,lung_opacity",
        help="Comma-separated XRV label slugs used for semantic supervision.",
    )
    parser.add_argument(
        "--xrv-semantic-factual-weight",
        type=float,
        default=1.0,
        help="Relative weight for matching factual mean-image XRV labels to the observed follow-up.",
    )
    parser.add_argument(
        "--xrv-semantic-response-weight",
        type=float,
        default=1.0,
        help=(
            "Relative weight for matching the predicted treatment contrast to observed "
            "treated baseline-to-follow-up label reductions."
        ),
    )
    parser.add_argument(
        "--xrv-semantic-response-gate-column",
        default="",
        help=(
            "Optional numeric CSV column carried into each batch. When set, the XRV "
            "response loss is applied only to treated examples with gate > 0."
        ),
    )
    parser.add_argument(
        "--xrv-semantic-null-gate-column",
        default="",
        help=(
            "Optional numeric CSV column carried into each batch. When set, examples "
            "with gate > 0 are penalized for positive generated treatment-response "
            "semantic reductions."
        ),
    )
    parser.add_argument(
        "--xrv-semantic-response-mode",
        choices=["observed_xrv", "positive_margin"],
        default="observed_xrv",
        help=(
            "observed_xrv matches observed image-label reduction; positive_margin "
            "uses report-derived response gates to require a positive generated "
            "fluid-label treatment contrast."
        ),
    )
    parser.add_argument(
        "--xrv-semantic-response-margin",
        type=float,
        default=0.0,
        help="Minimum generated XRV label reduction used in positive_margin mode.",
    )
    parser.add_argument(
        "--xrv-semantic-null-margin",
        type=float,
        default=0.0,
        help="Allowed generated XRV label reduction for null-gated examples.",
    )
    parser.add_argument(
        "--xrv-semantic-null-weight",
        type=float,
        default=0.0,
        help="Relative weight for specificity loss on null-gated examples.",
    )
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--skip-image-verify", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denorm(tensor):
    return tensor.detach().float().clamp(-1.0, 1.0).add(1.0).div(2.0)


def unit_interval(tensor):
    """Map model images from [-1, 1] to [0, 1] without detaching gradients."""
    return tensor.float().clamp(-1.0, 1.0).add(1.0).div(2.0)


def slugify_label(label):
    return re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")


def parse_label_slugs(labels):
    raw_labels = re.split(r"[,+:;|]+", str(labels))
    return [slugify_label(label) for label in raw_labels if slugify_label(label)]


def load_xrv_semantic_bundle(weights, labels, device):
    if not weights:
        return None
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights=weights).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    requested = parse_label_slugs(labels)
    available = {slugify_label(label): index for index, label in enumerate(model.pathologies)}
    missing = [label for label in requested if label not in available]
    if missing:
        raise ValueError(
            f"Requested XRV semantic labels are unavailable for {weights}: {missing}. "
            f"Available labels: {sorted(key for key in available if key)}"
        )
    indices = [available[label] for label in requested]
    resolution = int(getattr(model, "input_resolution", 224) or 224)
    return {
        "model": model,
        "labels": requested,
        "indices": indices,
        "resolution": resolution,
        "weights": weights,
    }


def xrv_scores(bundle, image):
    image_01 = unit_interval(image)
    if image_01.shape[-2:] != (bundle["resolution"], bundle["resolution"]):
        image_01 = F.interpolate(
            image_01,
            size=(bundle["resolution"], bundle["resolution"]),
            mode="bilinear",
            align_corners=False,
        )
    xrv_input = (2.0 * image_01 - 1.0) * 1024.0
    return bundle["model"](xrv_input)[:, bundle["indices"]].float()


def weighted_semantic_average(values, weights=None, label_weights=None):
    if label_weights is not None:
        values = values * label_weights.view(1, -1)
    values = values.mean(dim=1)
    if weights is not None:
        values = values * weights.view(-1).float()
    return values.mean()


def load_autoencoder(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    latent_channels = int(config.get("latent_channels", 4))
    model = CXRLatentAutoencoder(
        base_channels=int(config.get("base_channels", 64)),
        latent_channels=latent_channels,
    ).to(device)
    state = checkpoint.get("ema_model_state_dict") or checkpoint["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config, latent_channels


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
    state = checkpoint.get("ema_model_state_dict") or checkpoint["model_state_dict"]
    model.load_state_dict(state)
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


@torch.no_grad()
def encode_pair(autoencoder, batch):
    baseline_latent = autoencoder.encode(batch["x_0"], sample=False)
    target_latent = autoencoder.encode(batch["y"], sample=False)
    return baseline_latent, target_latent


@torch.no_grad()
def predict_bridge_image(bridge_bundle, batch, device, use_amp):
    treatment = apply_conditioning_mode(batch["a"], bridge_bundle["conditioning_mode"])
    bridge_model = bridge_bundle["model"]
    bridge = bridge_bundle["bridge"]
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        if bridge_bundle["method"] == "one_step":
            prediction = bridge.one_step_predict(
                bridge_model,
                batch["x_0"],
                treatment,
                batch["delta"],
            )
        elif bridge_bundle["method"] == "iterative":
            prediction = bridge.iterative_predict(
                bridge_model,
                batch["x_0"],
                treatment,
                batch["delta"],
                inference_steps=bridge_bundle["inference_steps"],
            )
        else:
            raise ValueError(f"Unsupported bridge method: {bridge_bundle['method']}")
    return prediction.float().clamp(-1.0, 1.0)


@torch.no_grad()
def build_latent_inputs(
    autoencoder,
    batch,
    bridge_bundle,
    device,
    use_amp,
    target_mode="target",
    residual_scale=1.0,
):
    baseline_latent, target_latent = encode_pair(autoencoder, batch)
    bridge_image = None
    bridge_latent = None
    start_latent = baseline_latent
    conditioning_latent = baseline_latent
    if bridge_bundle is not None:
        bridge_image = predict_bridge_image(bridge_bundle, batch, device, use_amp)
        bridge_latent = autoencoder.encode(bridge_image, sample=False)
        conditioning_latent = torch.cat([baseline_latent, bridge_latent], dim=1)
        start_latent = bridge_latent
    if target_mode == "delta":
        if residual_scale <= 0:
            raise ValueError("--residual-scale must be positive")
        diffusion_target = (target_latent - baseline_latent) / residual_scale
        start_latent = torch.zeros_like(diffusion_target)
    elif target_mode == "residual":
        if bridge_latent is None:
            raise ValueError("--target-mode residual requires --bridge-checkpoint")
        if residual_scale <= 0:
            raise ValueError("--residual-scale must be positive")
        diffusion_target = (target_latent - bridge_latent) / residual_scale
        start_latent = torch.zeros_like(diffusion_target)
    elif target_mode == "target":
        diffusion_target = target_latent
    else:
        raise ValueError(f"Unsupported target_mode={target_mode}")
    return conditioning_latent, diffusion_target, start_latent, bridge_image, bridge_latent


def weighted_latent_l1(prediction, target, weights=None):
    loss = (prediction - target).abs().mean(dim=(1, 2, 3))
    if weights is not None:
        loss = loss * weights.view(-1).float()
    return loss.mean()


def high_pass(image, kernel_size=5):
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("High-pass kernel must be an odd integer >= 3")
    smooth = F.avg_pool2d(image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return image - smooth


def weighted_image_l1(prediction, target, weights=None):
    loss = (prediction - target).abs().mean(dim=(1, 2, 3))
    if weights is not None:
        loss = loss * weights.view(-1).float()
    return loss.mean()


def decoded_mean_image_losses(autoencoder, mean_latent, target_latent, highpass_kernel=5, weights=None):
    """Image-space losses for the mean branch; gradients flow through mean_latent."""
    mean_image = autoencoder.decode(mean_latent)
    target_image = autoencoder.decode(target_latent.detach())
    image_l1 = weighted_image_l1(mean_image, target_image, weights=weights)
    highpass_l1 = weighted_image_l1(
        high_pass(mean_image, highpass_kernel),
        high_pass(target_image, highpass_kernel),
        weights=weights,
    )
    return image_l1, highpass_l1


def treatment_effect_highpass_loss(autoencoder, mu0_latent, tau_latent, kernel_size=5):
    """Keep the treatment effect from erasing high-frequency decoded anatomy."""
    mu0_image = autoencoder.decode(mu0_latent.detach())
    mu1_image = autoencoder.decode(mu0_latent.detach() + tau_latent)
    return (high_pass(mu1_image, kernel_size) - high_pass(mu0_image, kernel_size)).abs().mean()


def central_field_mask_like(image):
    height, width = image.shape[-2:]
    mask = torch.zeros_like(image)
    y0 = int(round(0.06 * height))
    y1 = int(round(0.94 * height))
    x0 = int(round(0.04 * width))
    x1 = int(round(0.96 * width))
    mask[..., y0:y1, x0:x1] = 1.0
    foreground = torch.sigmoid((image - 0.04) / 0.025)
    return mask * foreground


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


def anatomic_lung_prior_like(image):
    height, width = image.shape[-2:]
    y = torch.linspace(0.0, 1.0, height, device=image.device, dtype=image.dtype).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, width, device=image.device, dtype=image.dtype).view(1, 1, 1, width)
    left = ((x - 0.36) / 0.23).pow(2) + ((y - 0.50) / 0.38).pow(2)
    right = ((x - 0.64) / 0.23).pow(2) + ((y - 0.50) / 0.38).pow(2)
    prior = torch.maximum(torch.sigmoid((1.0 - left) / 0.08), torch.sigmoid((1.0 - right) / 0.08))
    superior_taper = torch.sigmoid((y - 0.12) / 0.04)
    inferior_taper = torch.sigmoid((0.88 - y) / 0.05)
    return prior * superior_taper * inferior_taper


def lung_proxy_mask(image, method="hybrid"):
    valid = central_field_mask_like(image)
    dark_field = soft_dark_field_mask(image)
    prior = anatomic_lung_prior_like(image) * valid
    if method == "dark-field":
        mask = dark_field * valid
    elif method == "anatomic":
        mask = prior
    else:
        mask = prior * (0.35 + 0.65 * dark_field.clamp(0.0, 1.0))
    mask = F.avg_pool2d(mask, kernel_size=7, stride=1, padding=3)
    return mask.clamp(0.0, 1.0)


def weighted_masked_l1(delta, mask, weights=None):
    loss = (delta * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1e-8)
    if weights is not None:
        loss = loss * weights.view(-1).float()
    return loss.mean()


def treatment_effect_localization_losses(
    autoencoder,
    baseline_image,
    mu0_latent,
    tau_latent,
    mask_method="hybrid",
    lung_mask=None,
    weights=None,
):
    """Penalize decoded treatment effect outside lung fields and at borders."""
    baseline_01 = unit_interval(baseline_image.detach())
    mu0_image = unit_interval(autoencoder.decode(mu0_latent.detach()))
    mu1_image = unit_interval(autoencoder.decode(mu0_latent.detach() + tau_latent))
    delta = (mu1_image - mu0_image).abs()
    if mask_method == "precomputed":
        if lung_mask is None:
            raise KeyError(
                "--treatment-effect-mask-method precomputed requires --lung-mask-root "
                "so each batch contains a lung_mask tensor"
            )
        lung = lung_mask.to(device=baseline_01.device, dtype=baseline_01.dtype)
        if tuple(lung.shape[-2:]) != tuple(baseline_01.shape[-2:]):
            lung = F.interpolate(
                lung,
                size=baseline_01.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        lung = F.avg_pool2d(lung.clamp(0.0, 1.0), kernel_size=7, stride=1, padding=3)
    else:
        lung = lung_proxy_mask(baseline_01, method=mask_method)
    nonlung = (central_field_mask_like(baseline_01) * (1.0 - lung)).clamp(0.0, 1.0)
    border = border_mask_like(baseline_01)
    return {
        "nonlung": weighted_masked_l1(delta, nonlung, weights=weights),
        "border": weighted_masked_l1(delta, border, weights=weights),
        "lung": weighted_masked_l1(delta, lung, weights=weights),
    }


def treatment_effect_r_learner_loss(
    target_latent,
    mu0_latent,
    tau_latent,
    treatment,
    propensity,
    weights=None,
    propensity_eps=0.02,
):
    """Orthogonalized treatment-effect loss in latent space.

    This is the latent analogue of the R-learner objective
        ||(Y - m(X)) - (A - e(X)) tau(X)||^2.
    Gradients are intentionally stopped through m(X)=mu0, so this term
    updates the treatment-response head rather than the nuisance outcome model.
    """
    propensity = propensity.float()
    if propensity_eps is not None and propensity_eps > 0:
        propensity = propensity.clamp(propensity_eps, 1.0 - propensity_eps)
    treatment_residual = treatment.float() - propensity
    treatment_residual = treatment_residual.view(-1, 1, 1, 1)
    outcome_residual = target_latent.detach() - mu0_latent.detach()
    predicted_residual = treatment_residual * tau_latent
    loss = (outcome_residual - predicted_residual).pow(2).mean(dim=(1, 2, 3))
    if weights is not None:
        loss = loss * weights.view(-1).float()
    return loss.mean()


def xrv_semantic_potential_losses(
    autoencoder,
    xrv_bundle,
    batch,
    mean_latent,
    target_latent,
    mu0_latent,
    tau_latent,
    weights=None,
    response_gate=None,
    null_gate=None,
    response_mode="observed_xrv",
    response_margin=0.0,
    null_margin=0.0,
):
    """Frozen-CXR-labeler supervision for potential-outcome mean fields.

    The factual term makes the decoded mean image match observed follow-up
    CXR labels. The response term is only applied to treated cases whose
    observed follow-up label decreased from baseline, and matches that
    observed reduction with the learned potential contrast phi(Y^0)-phi(Y^1).
    """
    mean_image = autoencoder.decode(mean_latent)
    mu0_image = autoencoder.decode(mu0_latent.detach())
    mu1_image = autoencoder.decode(mu0_latent.detach() + tau_latent)
    target_image = batch["y"].detach()
    baseline_image = batch["x_0"].detach()

    mean_scores = xrv_scores(xrv_bundle, mean_image)
    target_scores = xrv_scores(xrv_bundle, target_image).detach()
    baseline_scores = xrv_scores(xrv_bundle, baseline_image).detach()
    mu0_scores = xrv_scores(xrv_bundle, mu0_image).detach()
    mu1_scores = xrv_scores(xrv_bundle, mu1_image)

    factual_loss = weighted_semantic_average((mean_scores - target_scores).abs(), weights=weights)

    observed_reduction = (baseline_scores - target_scores).clamp_min(0.0)
    predicted_reduction = mu0_scores - mu1_scores
    treated = (batch["a"].view(-1) >= 0.5).float()
    if response_gate is not None:
        gate = (response_gate.view(-1, 1).float() > 0.0).float()
        response_weight = gate
        if response_mode == "positive_margin":
            margin = torch.as_tensor(
                float(response_margin),
                dtype=predicted_reduction.dtype,
                device=predicted_reduction.device,
            )
            response_error = F.relu(margin - predicted_reduction)
        else:
            response_weight = response_weight * (observed_reduction > 0.0).float()
            response_error = (predicted_reduction - observed_reduction).abs()
    else:
        response_weight = treated.view(-1, 1) * (observed_reduction > 0.0).float()
        response_error = (predicted_reduction - observed_reduction).abs()
    if weights is not None:
        response_weight = response_weight * weights.view(-1, 1).float()
    denom = response_weight.sum().clamp_min(1.0)
    response_loss = (response_error * response_weight).sum() / denom

    null_loss = predicted_reduction.new_tensor(0.0)
    if null_gate is not None:
        null_weight = (null_gate.view(-1, 1).float() > 0.0).float()
        allowed = torch.as_tensor(
            float(null_margin),
            dtype=predicted_reduction.dtype,
            device=predicted_reduction.device,
        )
        null_error = F.relu(predicted_reduction - allowed)
        if weights is not None:
            null_weight = null_weight * weights.view(-1, 1).float()
        null_denom = null_weight.sum().clamp_min(1.0)
        null_loss = (null_error * null_weight).sum() / null_denom

    return {
        "factual": factual_loss,
        "response": response_loss,
        "null": null_loss,
    }


def effective_sample_weights(batch, args):
    weights = batch.get("sample_weight")
    if weights is not None:
        weights = weights.view(-1).float()
    elif args.treated_loss_multiplier != 1.0:
        weights = torch.ones(batch["a"].shape[0], device=batch["a"].device)

    if weights is not None and args.treated_loss_multiplier != 1.0:
        treated = batch["a"].view(-1).float() > 0.5
        multiplier = torch.where(
            treated,
            torch.full_like(weights, float(args.treated_loss_multiplier)),
            torch.ones_like(weights),
        )
        weights = weights * multiplier
    return weights


def predict_mean_latent(
    mean_head,
    baseline_latent,
    batch,
    treatment_conditioning_mode="full",
    return_components=False,
):
    treatment = apply_conditioning_mode(batch["a"], treatment_conditioning_mode)
    if return_components and getattr(mean_head, "supports_potential_outcomes", False):
        mean_latent, mu0_latent, tau_latent = mean_head(
            baseline_latent,
            treatment,
            batch["delta"],
            return_components=True,
        )
        return mean_latent, {"mu0_latent": mu0_latent, "tau_latent": tau_latent}
    mean_latent = mean_head(baseline_latent, treatment, batch["delta"])
    return mean_latent, {}


def build_mean_residual_inputs(
    autoencoder,
    mean_head,
    batch,
    residual_scale,
    treatment_conditioning_mode="full",
):
    if residual_scale <= 0:
        raise ValueError("--residual-scale must be positive")
    with torch.no_grad():
        baseline_latent, target_latent = encode_pair(autoencoder, batch)
    mean_latent, components = predict_mean_latent(
        mean_head,
        baseline_latent,
        batch,
        treatment_conditioning_mode=treatment_conditioning_mode,
        return_components=True,
    )
    mean_anchor = mean_latent.detach()
    diffusion_target = (target_latent - mean_anchor) / residual_scale
    conditioning_latent = torch.cat([baseline_latent, mean_anchor], dim=1)
    start_latent = torch.zeros_like(diffusion_target)
    return conditioning_latent, diffusion_target, start_latent, mean_latent, target_latent, components


def build_dataloaders(args):
    split_frames = prepare_pair_splits(
        csv_path=args.csv_path,
        image_root=args.image_root,
        split_dir=args.split_dir,
        seed=args.seed,
        verify_readable=not args.skip_image_verify,
    )
    transform = build_transform(
        args.image_size,
        foreground_crop=args.foreground_crop,
        crop_threshold=args.crop_threshold,
        crop_min_content_fraction=args.crop_min_content_fraction,
        crop_margin_fraction=args.crop_margin_fraction,
    )
    weight_column = None
    if args.loss_weight_column and args.loss_weight_column.lower() != "none":
        weight_column = args.loss_weight_column
    propensity_column = None
    if (
        args.treatment_effect_r_loss_weight > 0
        and args.treatment_propensity_column
        and args.treatment_propensity_column.lower() != "none"
    ):
        propensity_column = args.treatment_propensity_column
    lung_mask_root = None
    if args.lung_mask_root and args.lung_mask_root.lower() != "none":
        lung_mask_root = args.lung_mask_root
    if args.treatment_effect_mask_method == "precomputed" and not lung_mask_root:
        raise ValueError("--treatment-effect-mask-method precomputed requires --lung-mask-root")
    extra_numeric_columns = []
    if args.xrv_semantic_response_gate_column:
        extra_numeric_columns.append(args.xrv_semantic_response_gate_column)
    if args.xrv_semantic_null_gate_column:
        extra_numeric_columns.append(args.xrv_semantic_null_gate_column)
    datasets = {
        split_name: CXRPairDataset(
            frame,
            args.image_root,
            transform,
            weight_column=weight_column,
            propensity_column=propensity_column,
            treatment_column=args.treatment_column,
            lung_mask_root=lung_mask_root,
            extra_numeric_columns=extra_numeric_columns,
        )
        for split_name, frame in split_frames.items()
    }
    pin_memory = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=args.num_workers > 0,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=args.num_workers > 0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=args.num_workers > 0,
        ),
    }
    return loaders, split_frames


@torch.no_grad()
def evaluate_loss(
    model,
    mean_head,
    autoencoder,
    ddpm,
    loader,
    device,
    use_amp,
    bridge_bundle,
    args,
    xrv_bundle=None,
):
    model.eval()
    if mean_head is not None:
        mean_head.eval()
    losses = []
    for batch in tqdm(loader, desc="latent val loss"):
        batch = move_batch_to_device(batch, device)
        if args.target_mode == "mean_residual":
            conditioning_latent, diffusion_target, _, mean_latent, target_latent, mean_components = (
                build_mean_residual_inputs(
                    autoencoder,
                    mean_head,
                    batch,
                    args.residual_scale,
                    treatment_conditioning_mode=args.treatment_conditioning_mode,
                )
            )
        else:
            conditioning_latent, diffusion_target, _, _, _ = build_latent_inputs(
                autoencoder,
                batch,
                bridge_bundle,
                device,
                use_amp,
                target_mode=args.target_mode,
                residual_scale=args.residual_scale,
            )
        timesteps = torch.randint(
            0,
            ddpm.num_timesteps,
            (diffusion_target.shape[0],),
            device=device,
        )
        weights = effective_sample_weights(batch, args)
        treatment = apply_conditioning_mode(batch["a"], args.treatment_conditioning_mode)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            diffusion_loss = ddpm.p_losses(
                model,
                diffusion_target,
                timesteps,
                conditioning_latent,
                treatment,
                batch["delta"],
                weights=weights,
            )
            loss = args.diffusion_loss_weight * diffusion_loss
            if args.target_mode == "mean_residual":
                loss = loss + args.mean_loss_weight * weighted_latent_l1(
                    mean_latent,
                    target_latent,
                    weights=weights,
                )
                if args.mean_image_l1_weight > 0 or args.mean_highpass_weight > 0:
                    mean_image_l1, mean_highpass_l1 = decoded_mean_image_losses(
                        autoencoder,
                        mean_latent,
                        target_latent,
                        highpass_kernel=args.mean_highpass_kernel,
                        weights=weights,
                    )
                    if args.mean_image_l1_weight > 0:
                        loss = loss + args.mean_image_l1_weight * mean_image_l1
                    if args.mean_highpass_weight > 0:
                        loss = loss + args.mean_highpass_weight * mean_highpass_l1
                tau_latent = mean_components.get("tau_latent")
                mu0_latent = mean_components.get("mu0_latent")
                if tau_latent is not None and args.treatment_effect_l1_weight > 0:
                    loss = loss + args.treatment_effect_l1_weight * weighted_latent_l1(
                        tau_latent,
                        torch.zeros_like(tau_latent),
                        weights=weights,
                    )
                if (
                    tau_latent is not None
                    and mu0_latent is not None
                    and args.treatment_effect_r_loss_weight > 0
                ):
                    if "propensity" not in batch:
                        raise KeyError(
                            "--treatment-effect-r-loss-weight requires "
                            "--treatment-propensity-column to be present in the dataset"
                        )
                    loss = loss + args.treatment_effect_r_loss_weight * treatment_effect_r_learner_loss(
                        target_latent,
                        mu0_latent,
                        tau_latent,
                        batch["a"],
                        batch["propensity"],
                        weights=weights,
                        propensity_eps=args.treatment_propensity_eps,
                    )
                if (
                    tau_latent is not None
                    and mu0_latent is not None
                    and args.treatment_effect_highpass_weight > 0
                ):
                    loss = loss + args.treatment_effect_highpass_weight * treatment_effect_highpass_loss(
                        autoencoder,
                        mu0_latent,
                        tau_latent,
                        kernel_size=args.treatment_effect_highpass_kernel,
                    )
                if (
                    tau_latent is not None
                    and mu0_latent is not None
                    and (
                        args.treatment_effect_nonlung_weight > 0
                        or args.treatment_effect_border_weight > 0
                    )
                ):
                    localization_losses = treatment_effect_localization_losses(
                        autoencoder,
                        batch["x_0"],
                        mu0_latent,
                        tau_latent,
                        mask_method=args.treatment_effect_mask_method,
                        lung_mask=batch.get("lung_mask"),
                        weights=weights,
                    )
                    if args.treatment_effect_nonlung_weight > 0:
                        loss = (
                            loss
                            + args.treatment_effect_nonlung_weight
                            * localization_losses["nonlung"]
                        )
                    if args.treatment_effect_border_weight > 0:
                        loss = (
                            loss
                            + args.treatment_effect_border_weight
                            * localization_losses["border"]
                        )
                if (
                    xrv_bundle is not None
                    and tau_latent is not None
                    and mu0_latent is not None
                    and args.xrv_semantic_loss_weight > 0
                ):
                    semantic_losses = xrv_semantic_potential_losses(
                        autoencoder,
                        xrv_bundle,
                        batch,
                        mean_latent,
                        target_latent,
                        mu0_latent,
                        tau_latent,
                        weights=weights,
                        response_gate=batch.get(args.xrv_semantic_response_gate_column)
                        if args.xrv_semantic_response_gate_column
                        else None,
                        null_gate=batch.get(args.xrv_semantic_null_gate_column)
                        if args.xrv_semantic_null_gate_column
                        else None,
                        response_mode=args.xrv_semantic_response_mode,
                        response_margin=args.xrv_semantic_response_margin,
                        null_margin=args.xrv_semantic_null_margin,
                    )
                    loss = loss + args.xrv_semantic_loss_weight * (
                        args.xrv_semantic_factual_weight * semantic_losses["factual"]
                        + args.xrv_semantic_response_weight * semantic_losses["response"]
                        + args.xrv_semantic_null_weight * semantic_losses["null"]
                    )
        losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def generate_batch(autoencoder, model, mean_head, ddpm, batch, device, args, bridge_bundle):
    batch = move_batch_to_device(batch, device)
    reference_image = None
    bridge_latent = None
    mean_latent = None
    if args.target_mode == "mean_residual":
        conditioning_latent, _, start_latent, mean_latent, _, _ = build_mean_residual_inputs(
            autoencoder,
            mean_head,
            batch,
            args.residual_scale,
            treatment_conditioning_mode=args.treatment_conditioning_mode,
        )
        reference_image = autoencoder.decode(mean_latent)
    else:
        conditioning_latent, _, start_latent, bridge_image, bridge_latent = build_latent_inputs(
            autoencoder,
            batch,
            bridge_bundle,
            device,
            not args.disable_amp and device.type == "cuda",
            target_mode=args.target_mode,
            residual_scale=args.residual_scale,
        )
        reference_image = bridge_image
    sampled_latent = ddpm.ddim_sample(
        model,
        conditioning_latent,
        apply_conditioning_mode(batch["a"], args.treatment_conditioning_mode),
        batch["delta"],
        start_timestep=args.sample_start_timestep,
        steps=args.sample_steps,
        start_latent=start_latent,
    )
    if args.target_mode == "delta":
        baseline_latent = conditioning_latent[:, : sampled_latent.shape[1]]
        generated_latent = baseline_latent + args.residual_scale * sampled_latent
    elif args.target_mode == "residual":
        if bridge_latent is None:
            raise ValueError("Residual sampling requires a bridge latent")
        generated_latent = bridge_latent + args.residual_scale * sampled_latent
    elif args.target_mode == "mean_residual":
        generated_latent = mean_latent + args.residual_scale * sampled_latent
    else:
        generated_latent = sampled_latent
    generated = autoencoder.decode(generated_latent)
    return batch, generated, reference_image


def save_sample_panel(autoencoder, model, mean_head, ddpm, batch, device, args, output_path, bridge_bundle):
    model.eval()
    if mean_head is not None:
        mean_head.eval()
    batch, generated, reference_image = generate_batch(
        autoencoder,
        model,
        mean_head,
        ddpm,
        batch,
        device,
        args,
        bridge_bundle,
    )
    generated = generated.cpu()
    if reference_image is not None:
        reference_image = reference_image.cpu()
    batch_size = min(args.sample_count, generated.shape[0])
    rows = 4 if reference_image is not None else 3
    fig, axes = plt.subplots(rows, batch_size, figsize=(4 * batch_size, 3 * rows))
    if batch_size == 1:
        axes = axes.reshape(rows, 1)
    for index in range(batch_size):
        baseline = denorm(batch["x_0"][index].cpu())[0]
        target = denorm(batch["y"][index].cpu())[0]
        prediction = denorm(generated[index])[0]
        axes[0, index].imshow(baseline, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, index].set_title(
            f"Baseline a={int(batch['a'][index].item())}, dt={batch['hours_diff'][index].item():.0f}h"
        )
        row = 1
        if reference_image is not None:
            reference_prediction = denorm(reference_image[index])[0]
            axes[row, index].imshow(reference_prediction, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, index].set_title("Mean" if args.target_mode == "mean_residual" else "Bridge")
            row += 1
        axes[row, index].imshow(target, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, index].set_title("Ground truth")
        row += 1
        axes[row, index].imshow(prediction, cmap="gray", vmin=0.0, vmax=1.0)
        if args.target_mode == "delta":
            title = "Baseline + delta DDPM"
        elif args.target_mode == "residual":
            title = "Bridge + residual DDPM"
        elif args.target_mode == "mean_residual":
            title = "Mean + residual DDPM"
        else:
            title = "Latent DDPM"
        axes[row, index].set_title(title)
        for row in range(rows):
            axes[row, index].axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_checkpoint(
    path,
    epoch,
    val_loss,
    model,
    ema,
    optimizer,
    scaler,
    config,
    mean_head=None,
    mean_ema=None,
):
    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "config": config,
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema.ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
    }
    if mean_head is not None:
        checkpoint["mean_model_state_dict"] = mean_head.state_dict()
        checkpoint["mean_ema_model_state_dict"] = mean_ema.ema_model.state_dict()
    torch.save(checkpoint, path)


def load_init_checkpoint(path, model, allow_partial=False):
    if not path:
        return None
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("ema_model_state_dict") or checkpoint["model_state_dict"]
    if not allow_partial:
        model.load_state_dict(state, strict=True)
        return checkpoint

    target_state = model.state_dict()
    load_state = {}
    skipped = []
    expanded = []
    for key, value in state.items():
        if key not in target_state:
            skipped.append(key)
            continue
        target_value = target_state[key]
        if value.shape == target_value.shape:
            load_state[key] = value
            continue
        if (
            key == "init.weight"
            and value.ndim == target_value.ndim == 4
            and value.shape[0] == target_value.shape[0]
            and value.shape[2:] == target_value.shape[2:]
            and value.shape[1] <= target_value.shape[1]
        ):
            expanded_weight = target_value.clone()
            expanded_weight.zero_()
            expanded_weight[:, : value.shape[1], :, :] = value
            load_state[key] = expanded_weight
            expanded.append(f"{key}: {tuple(value.shape)} -> {tuple(target_value.shape)}")
            continue
        skipped.append(f"{key}: {tuple(value.shape)} -> {tuple(target_value.shape)}")
    target_state.update(load_state)
    model.load_state_dict(target_state, strict=True)
    print(
        "Loaded partial latent DDPM init: "
        f"{len(load_state)} tensors, expanded={expanded}, skipped={len(skipped)}"
    )
    return checkpoint


def load_mean_init_checkpoint(path, mean_head, allow_partial=False):
    if not path:
        return None
    checkpoint = torch.load(path, map_location="cpu")
    state = (
        checkpoint.get("mean_ema_model_state_dict")
        or checkpoint.get("mean_model_state_dict")
    )
    if state is None:
        return checkpoint
    init_mean_architecture = checkpoint.get("config", {}).get("mean_architecture", "standard")
    if (
        hasattr(mean_head, "progression")
        and isinstance(mean_head.progression, LatentMeanPredictor)
        and init_mean_architecture == "standard"
    ):
        mean_head.progression.load_state_dict(state, strict=True)
        print("Loaded standard mean predictor init into anchored progression branch")
        return checkpoint
    if not allow_partial:
        mean_head.load_state_dict(state, strict=True)
        return checkpoint

    target_state = mean_head.state_dict()
    load_state = {}
    skipped = []
    for key, value in state.items():
        if key in target_state and value.shape == target_state[key].shape:
            load_state[key] = value
        else:
            skipped.append(key)
    target_state.update(load_state)
    mean_head.load_state_dict(target_state, strict=True)
    missing = sorted(set(mean_head.state_dict()) - set(load_state))
    print(
        "Loaded partial mean predictor init: "
        f"{len(load_state)} tensors, skipped={len(skipped)}, missing_initialized={len(missing)}"
    )
    if skipped:
        print(f"Skipped mean tensors: {skipped[:8]}{' ...' if len(skipped) > 8 else ''}")
    return checkpoint


def freeze_potential_progression(mean_head):
    if not getattr(mean_head, "supports_potential_outcomes", False):
        raise ValueError("--freeze-potential-progression requires --mean-architecture potential_outcome")
    frozen = 0
    trainable = 0
    for name, parameter in mean_head.named_parameters():
        if name.startswith("tau_"):
            parameter.requires_grad_(True)
            trainable += parameter.numel()
        else:
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    return frozen, trainable


def freeze_module(module):
    frozen = 0
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        frozen += parameter.numel()
    return frozen


def build_fixed_batch(split_frame, args):
    sample_count = min(args.sample_count, len(split_frame))
    weight_column = None
    if args.loss_weight_column and args.loss_weight_column.lower() != "none":
        weight_column = args.loss_weight_column
    propensity_column = None
    if (
        args.treatment_effect_r_loss_weight > 0
        and args.treatment_propensity_column
        and args.treatment_propensity_column.lower() != "none"
    ):
        propensity_column = args.treatment_propensity_column
    lung_mask_root = None
    if args.lung_mask_root and args.lung_mask_root.lower() != "none":
        lung_mask_root = args.lung_mask_root
    if args.treatment_effect_mask_method == "precomputed" and not lung_mask_root:
        raise ValueError("--treatment-effect-mask-method precomputed requires --lung-mask-root")
    extra_numeric_columns = []
    if args.xrv_semantic_response_gate_column:
        extra_numeric_columns.append(args.xrv_semantic_response_gate_column)
    if args.xrv_semantic_null_gate_column:
        extra_numeric_columns.append(args.xrv_semantic_null_gate_column)
    dataset = CXRPairDataset(
        split_frame.iloc[:sample_count].copy(),
        args.image_root,
        build_transform(
            args.image_size,
            foreground_crop=args.foreground_crop,
            crop_threshold=args.crop_threshold,
            crop_min_content_fraction=args.crop_min_content_fraction,
            crop_margin_fraction=args.crop_margin_fraction,
        ),
        weight_column=weight_column,
        propensity_column=propensity_column,
        treatment_column=args.treatment_column,
        lung_mask_root=lung_mask_root,
        extra_numeric_columns=extra_numeric_columns,
    )
    loader = DataLoader(dataset, batch_size=sample_count, shuffle=False, num_workers=0)
    return next(iter(loader))


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    sample_dir = Path(args.sample_dir) if args.sample_dir else output_dir / "samples"
    save_path = Path(args.save_path) if args.save_path else output_dir / "best_model.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.disable_amp) and device.type == "cuda"
    autoencoder, autoencoder_config, detected_latent_channels = load_autoencoder(
        args.autoencoder_checkpoint,
        device,
    )
    xrv_bundle = None
    if args.xrv_semantic_loss_weight > 0:
        if not args.xrv_semantic_weights:
            raise ValueError("--xrv-semantic-loss-weight requires --xrv-semantic-weights")
        xrv_bundle = load_xrv_semantic_bundle(
            args.xrv_semantic_weights,
            args.xrv_semantic_labels,
            device,
        )
        print(
            "Frozen XRV semantic supervision enabled: "
            f"weights={xrv_bundle['weights']} labels={xrv_bundle['labels']}"
        )
    bridge_bundle = load_bridge(
        args.bridge_checkpoint,
        device,
        method_override=args.bridge_method,
        inference_steps_override=args.bridge_inference_steps,
    )
    latent_channels = args.latent_channels or detected_latent_channels
    loaders, split_frames = build_dataloaders(args)
    conditioning_latents = 2 if bridge_bundle is not None or args.target_mode == "mean_residual" else 1
    model = LatentConditionalUNet(
        latent_channels=latent_channels,
        conditioning_latents=conditioning_latents,
        base_channels=args.latent_base_channels,
    ).to(device)
    mean_head = None
    if args.target_mode == "mean_residual":
        if args.mean_architecture == "potential_outcome":
            mean_head = LatentPotentialOutcomeMeanPredictor(
                latent_channels=latent_channels,
                base_channels=args.mean_base_channels,
                treatment_effect_scale=args.treatment_effect_scale,
            ).to(device)
        elif args.mean_architecture == "anchored_potential_outcome":
            mean_head = LatentAnchoredPotentialOutcomeMeanPredictor(
                latent_channels=latent_channels,
                base_channels=args.mean_base_channels,
                treatment_effect_scale=args.treatment_effect_scale,
            ).to(device)
        else:
            mean_head = LatentMeanPredictor(
                latent_channels=latent_channels,
                base_channels=args.mean_base_channels,
            ).to(device)
    init_checkpoint = load_init_checkpoint(
        args.init_checkpoint,
        model,
        allow_partial=args.allow_partial_init,
    )
    if mean_head is not None and init_checkpoint is not None:
        init_mean_architecture = init_checkpoint.get("config", {}).get(
            "mean_architecture",
            "standard",
        )
        allow_partial_mean_init = (
            args.allow_partial_init
            or args.mean_architecture != init_mean_architecture
        )
        load_mean_init_checkpoint(
            args.init_checkpoint,
            mean_head,
            allow_partial=allow_partial_mean_init,
        )
        print(f"Loaded mean predictor init from {args.init_checkpoint}")
    if args.freeze_potential_progression:
        if mean_head is None:
            raise ValueError("--freeze-potential-progression requires --target-mode mean_residual")
        frozen, trainable = freeze_potential_progression(mean_head)
        print(
            "Frozen potential-outcome progression branch: "
            f"{frozen:,} parameters frozen, {trainable:,} tau parameters trainable"
        )
    if args.freeze_ddpm:
        frozen = freeze_module(model)
        print(f"Frozen latent DDPM residual model: {frozen:,} parameters frozen")
    print(f"Latent DDPM parameters: {sum(p.numel() for p in model.parameters()):,}")
    if mean_head is not None:
        print(f"Latent mean predictor parameters: {sum(p.numel() for p in mean_head.parameters()):,}")
    if bridge_bundle is not None:
        print(
            "Bridge conditioning enabled: "
            f"{bridge_bundle['checkpoint_path']} "
            f"(method={bridge_bundle['method']}, "
            f"inference_steps={bridge_bundle['inference_steps']}, "
            f"conditioning_mode={bridge_bundle['conditioning_mode']})"
        )
    ema = ExponentialMovingAverage(model, args.ema_decay)
    if init_checkpoint is not None and "ema_model_state_dict" in init_checkpoint:
        load_init_checkpoint(
            args.init_checkpoint,
            ema.ema_model,
            allow_partial=args.allow_partial_init,
        )
    mean_ema = ExponentialMovingAverage(mean_head, args.ema_decay) if mean_head is not None else None
    ddpm = LatentDDPM(args.num_timesteps, args.beta_schedule).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if mean_head is not None:
        trainable_parameters.extend(
            parameter for parameter in mean_head.parameters() if parameter.requires_grad
        )
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after freeze options")
    optimizer = AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    fixed_batch = build_fixed_batch(split_frames["val"], args)

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "use_amp": use_amp,
            "latent_channels": latent_channels,
            "conditioning_latents": conditioning_latents,
            "autoencoder_config": autoencoder_config,
            "bridge_conditioned": bridge_bundle is not None,
            "target_mode": args.target_mode,
            "residual_scale": args.residual_scale,
            "bridge_config": bridge_bundle["config"] if bridge_bundle is not None else None,
            "bridge_method_resolved": bridge_bundle["method"] if bridge_bundle is not None else None,
            "bridge_inference_steps_resolved": (
                bridge_bundle["inference_steps"] if bridge_bundle is not None else None
            ),
            "save_path": str(save_path),
            "sample_dir": str(sample_dir),
        }
    )
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        if mean_head is not None:
            mean_head.train()
        train_losses = []
        train_diffusion_losses = []
        train_mean_losses = []
        train_mean_image_losses = []
        train_mean_highpass_losses = []
        train_tau_losses = []
        train_tau_r_losses = []
        train_tau_highpass_losses = []
        train_tau_nonlung_losses = []
        train_tau_border_losses = []
        train_tau_lung_losses = []
        train_xrv_factual_losses = []
        train_xrv_response_losses = []
        train_xrv_null_losses = []
        for batch in tqdm(loaders["train"], desc=f"LDM epoch {epoch}/{args.epochs}"):
            batch = move_batch_to_device(batch, device)
            if args.target_mode == "mean_residual":
                conditioning_latent, diffusion_target, _, mean_latent, target_latent, mean_components = (
                    build_mean_residual_inputs(
                        autoencoder,
                        mean_head,
                        batch,
                        args.residual_scale,
                        treatment_conditioning_mode=args.treatment_conditioning_mode,
                    )
                )
            else:
                with torch.no_grad():
                    conditioning_latent, diffusion_target, _, _, _ = build_latent_inputs(
                        autoencoder,
                        batch,
                        bridge_bundle,
                        device,
                        use_amp,
                        target_mode=args.target_mode,
                        residual_scale=args.residual_scale,
                    )
            timesteps = torch.randint(
                0,
                ddpm.num_timesteps,
                (diffusion_target.shape[0],),
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            treatment = apply_conditioning_mode(batch["a"], args.treatment_conditioning_mode)
            weights = effective_sample_weights(batch, args)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                diffusion_loss = ddpm.p_losses(
                    model,
                    diffusion_target,
                    timesteps,
                    conditioning_latent,
                    treatment,
                    batch["delta"],
                    weights=weights,
                )
                loss = args.diffusion_loss_weight * diffusion_loss
                mean_loss = None
                mean_image_l1 = None
                mean_highpass_l1 = None
                tau_l1_loss = None
                tau_r_loss = None
                tau_highpass_loss = None
                tau_nonlung_loss = None
                tau_border_loss = None
                tau_lung_loss = None
                xrv_factual_loss = None
                xrv_response_loss = None
                xrv_null_loss = None
                if args.target_mode == "mean_residual":
                    mean_loss = weighted_latent_l1(
                        mean_latent,
                        target_latent,
                        weights=weights,
                    )
                    loss = diffusion_loss + args.mean_loss_weight * mean_loss
                    if args.mean_image_l1_weight > 0 or args.mean_highpass_weight > 0:
                        mean_image_l1, mean_highpass_l1 = decoded_mean_image_losses(
                            autoencoder,
                            mean_latent,
                            target_latent,
                            highpass_kernel=args.mean_highpass_kernel,
                            weights=weights,
                        )
                    if mean_image_l1 is not None and args.mean_image_l1_weight > 0:
                        loss = loss + args.mean_image_l1_weight * mean_image_l1
                    if mean_highpass_l1 is not None and args.mean_highpass_weight > 0:
                        loss = loss + args.mean_highpass_weight * mean_highpass_l1
                    tau_latent = mean_components.get("tau_latent")
                    if tau_latent is not None:
                        tau_l1_loss = weighted_latent_l1(
                            tau_latent,
                            torch.zeros_like(tau_latent),
                            weights=weights,
                        )
                    if tau_l1_loss is not None and args.treatment_effect_l1_weight > 0:
                        loss = loss + args.treatment_effect_l1_weight * tau_l1_loss
                    mu0_latent = mean_components.get("mu0_latent")
                    if (
                        tau_latent is not None
                        and mu0_latent is not None
                        and args.treatment_effect_r_loss_weight > 0
                    ):
                        if "propensity" not in batch:
                            raise KeyError(
                                "--treatment-effect-r-loss-weight requires "
                                "--treatment-propensity-column to be present in the dataset"
                            )
                        tau_r_loss = treatment_effect_r_learner_loss(
                            target_latent,
                            mu0_latent,
                            tau_latent,
                            batch["a"],
                            batch["propensity"],
                            weights=weights,
                            propensity_eps=args.treatment_propensity_eps,
                        )
                        loss = loss + args.treatment_effect_r_loss_weight * tau_r_loss
                    if (
                        tau_latent is not None
                        and mu0_latent is not None
                        and args.treatment_effect_highpass_weight > 0
                    ):
                        tau_highpass_loss = treatment_effect_highpass_loss(
                            autoencoder,
                            mu0_latent,
                            tau_latent,
                            kernel_size=args.treatment_effect_highpass_kernel,
                        )
                        loss = loss + args.treatment_effect_highpass_weight * tau_highpass_loss
                    if (
                        tau_latent is not None
                        and mu0_latent is not None
                        and (
                            args.treatment_effect_nonlung_weight > 0
                            or args.treatment_effect_border_weight > 0
                        )
                    ):
                        localization_losses = treatment_effect_localization_losses(
                            autoencoder,
                            batch["x_0"],
                            mu0_latent,
                            tau_latent,
                            mask_method=args.treatment_effect_mask_method,
                            lung_mask=batch.get("lung_mask"),
                            weights=weights,
                        )
                        tau_nonlung_loss = localization_losses["nonlung"]
                        tau_border_loss = localization_losses["border"]
                        tau_lung_loss = localization_losses["lung"]
                        if args.treatment_effect_nonlung_weight > 0:
                            loss = loss + args.treatment_effect_nonlung_weight * tau_nonlung_loss
                        if args.treatment_effect_border_weight > 0:
                            loss = loss + args.treatment_effect_border_weight * tau_border_loss
                    if (
                        xrv_bundle is not None
                        and tau_latent is not None
                        and mu0_latent is not None
                        and args.xrv_semantic_loss_weight > 0
                    ):
                        semantic_losses = xrv_semantic_potential_losses(
                            autoencoder,
                            xrv_bundle,
                            batch,
                            mean_latent,
                            target_latent,
                            mu0_latent,
                            tau_latent,
                            weights=weights,
                            response_gate=batch.get(args.xrv_semantic_response_gate_column)
                            if args.xrv_semantic_response_gate_column
                            else None,
                            null_gate=batch.get(args.xrv_semantic_null_gate_column)
                            if args.xrv_semantic_null_gate_column
                            else None,
                            response_mode=args.xrv_semantic_response_mode,
                            response_margin=args.xrv_semantic_response_margin,
                            null_margin=args.xrv_semantic_null_margin,
                        )
                        xrv_factual_loss = semantic_losses["factual"]
                        xrv_response_loss = semantic_losses["response"]
                        xrv_null_loss = semantic_losses["null"]
                        loss = loss + args.xrv_semantic_loss_weight * (
                            args.xrv_semantic_factual_weight * xrv_factual_loss
                            + args.xrv_semantic_response_weight * xrv_response_loss
                            + args.xrv_semantic_null_weight * xrv_null_loss
                        )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            if mean_ema is not None:
                mean_ema.update(mean_head)
            train_losses.append(float(loss.item()))
            train_diffusion_losses.append(float(diffusion_loss.item()))
            if mean_loss is not None:
                train_mean_losses.append(float(mean_loss.item()))
            if mean_image_l1 is not None:
                train_mean_image_losses.append(float(mean_image_l1.item()))
            if mean_highpass_l1 is not None:
                train_mean_highpass_losses.append(float(mean_highpass_l1.item()))
            if tau_l1_loss is not None:
                train_tau_losses.append(float(tau_l1_loss.item()))
            if tau_r_loss is not None:
                train_tau_r_losses.append(float(tau_r_loss.item()))
            if tau_highpass_loss is not None:
                train_tau_highpass_losses.append(float(tau_highpass_loss.item()))
            if tau_nonlung_loss is not None:
                train_tau_nonlung_losses.append(float(tau_nonlung_loss.item()))
            if tau_border_loss is not None:
                train_tau_border_losses.append(float(tau_border_loss.item()))
            if tau_lung_loss is not None:
                train_tau_lung_losses.append(float(tau_lung_loss.item()))
            if xrv_factual_loss is not None:
                train_xrv_factual_losses.append(float(xrv_factual_loss.item()))
            if xrv_response_loss is not None:
                train_xrv_response_losses.append(float(xrv_response_loss.item()))
            if xrv_null_loss is not None:
                train_xrv_null_losses.append(float(xrv_null_loss.item()))

        val_loss = evaluate_loss(
            ema.ema_model,
            mean_ema.ema_model if mean_ema is not None else None,
            autoencoder,
            ddpm,
            loaders["val"],
            device,
            use_amp,
            bridge_bundle,
            args,
            xrv_bundle=xrv_bundle,
        )
        row = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
            "train_diffusion_loss": sum(train_diffusion_losses) / max(len(train_diffusion_losses), 1),
            "val_loss": val_loss,
        }
        if train_mean_losses:
            row["train_mean_l1"] = sum(train_mean_losses) / len(train_mean_losses)
        if train_mean_image_losses:
            row["train_mean_image_l1"] = sum(train_mean_image_losses) / len(train_mean_image_losses)
        if train_mean_highpass_losses:
            row["train_mean_highpass_l1"] = (
                sum(train_mean_highpass_losses) / len(train_mean_highpass_losses)
            )
        if train_tau_losses:
            row["train_tau_l1"] = sum(train_tau_losses) / len(train_tau_losses)
        if train_tau_r_losses:
            row["train_tau_r_learner_l2"] = sum(train_tau_r_losses) / len(train_tau_r_losses)
        if train_tau_highpass_losses:
            row["train_tau_highpass_l1"] = (
                sum(train_tau_highpass_losses) / len(train_tau_highpass_losses)
            )
        if train_tau_nonlung_losses:
            row["train_tau_nonlung_image_l1"] = (
                sum(train_tau_nonlung_losses) / len(train_tau_nonlung_losses)
            )
        if train_tau_border_losses:
            row["train_tau_border_image_l1"] = (
                sum(train_tau_border_losses) / len(train_tau_border_losses)
            )
        if train_tau_lung_losses:
            row["train_tau_lung_image_l1"] = (
                sum(train_tau_lung_losses) / len(train_tau_lung_losses)
            )
        if train_xrv_factual_losses:
            row["train_xrv_factual_l1"] = (
                sum(train_xrv_factual_losses) / len(train_xrv_factual_losses)
            )
        if train_xrv_response_losses:
            row["train_xrv_response_l1"] = (
                sum(train_xrv_response_losses) / len(train_xrv_response_losses)
            )
        if train_xrv_null_losses:
            row["train_xrv_null_l1"] = (
                sum(train_xrv_null_losses) / len(train_xrv_null_losses)
            )
        history.append(row)
        (output_dir / "latent_training_history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(row, indent=2))

        save_sample_panel(
            autoencoder,
            ema.ema_model,
            mean_ema.ema_model if mean_ema is not None else None,
            ddpm,
            fixed_batch,
            device,
            args,
            sample_dir / f"epoch_{epoch:03d}_val_panel.png",
            bridge_bundle,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                save_path,
                epoch,
                val_loss,
                model,
                ema,
                optimizer,
                scaler,
                config,
                mean_head=mean_head,
                mean_ema=mean_ema,
            )
            print(f"Saved best latent DDPM checkpoint to {save_path}")

        save_checkpoint(
            output_dir / "last_model.pt",
            epoch,
            val_loss,
            model,
            ema,
            optimizer,
            scaler,
            config,
            mean_head=mean_head,
            mean_ema=mean_ema,
        )

        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch}.pt",
                epoch,
                val_loss,
                model,
                ema,
                optimizer,
                scaler,
                config,
                mean_head=mean_head,
                mean_ema=mean_ema,
            )


if __name__ == "__main__":
    main()
