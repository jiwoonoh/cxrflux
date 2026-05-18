#!/usr/bin/env python3
"""Evaluate whether treatment-conditioned changes are anatomically localized."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluate_latent_diffusion import (
    border_mask_like,
    denorm,
    load_latent_checkpoint,
    prepare_eval_data,
    sample_images,
    soft_dark_field_mask,
)
from experiment_utils import move_batch_to_device
from mask_utils import load_lung_mask_batch


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute lung-localized treatment-effect diagnostics for latent "
            "CXR prediction models."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--split-dir", default=None)
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
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--sample-start-timestep", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mask-method",
        choices=["hybrid", "dark-field", "anatomic", "precomputed"],
        default="hybrid",
    )
    parser.add_argument(
        "--lung-mask-root",
        default="",
        help=(
            "Directory containing cached baseline lung masks keyed by cxr_0 DICOM id. "
            "Required when --mask-method precomputed is used."
        ),
    )
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--skip-image-verify", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean(image, mask):
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(EPS)
    return (image * mask).sum(dim=(1, 2, 3)) / denom


def masked_l1(left, right, mask):
    return masked_mean((left - right).abs(), mask)


def masked_sum(image, mask):
    return (image * mask).sum(dim=(1, 2, 3))


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


def lower_lung_roi_like(image):
    height, width = image.shape[-2:]
    mask = torch.zeros_like(image)
    y0 = int(round(0.48 * height))
    y1 = int(round(0.88 * height))
    x0 = int(round(0.10 * width))
    x1 = int(round(0.90 * width))
    mask[..., y0:y1, x0:x1] = 1.0
    return mask


def anatomic_lung_prior_like(image):
    """Soft two-ellipse lung prior in normalized image coordinates."""
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
    """Return a soft lung-field proxy mask in [0, 1]."""
    valid = central_field_mask_like(image)
    dark_field = soft_dark_field_mask(image)
    prior = anatomic_lung_prior_like(image) * valid
    if method == "dark-field":
        mask = dark_field * valid
    elif method == "anatomic":
        mask = prior
    else:
        # The prior prevents very opaque lungs from vanishing; the dark-field
        # term prevents the proxy from assigning too much weight to mediastinum.
        mask = prior * (0.35 + 0.65 * dark_field.clamp(0.0, 1.0))
    mask = F.avg_pool2d(mask, kernel_size=7, stride=1, padding=3)
    return mask.clamp(0.0, 1.0)


def outside_lung_mask(image, lung_mask):
    valid = central_field_mask_like(image)
    return (valid * (1.0 - lung_mask)).clamp(0.0, 1.0)


def resolve_lung_mask(args, batch, baseline):
    if args.mask_method != "precomputed":
        return lung_proxy_mask(baseline, method=args.mask_method)
    if "lung_mask" in batch:
        lung = batch["lung_mask"].to(device=baseline.device, dtype=baseline.dtype)
        if tuple(lung.shape[-2:]) != tuple(baseline.shape[-2:]):
            lung = F.interpolate(
                lung,
                size=baseline.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return lung.clamp(0.0, 1.0)
    if not args.lung_mask_root or args.lung_mask_root.lower() == "none":
        raise ValueError("--mask-method precomputed requires --lung-mask-root")
    return load_lung_mask_batch(
        args.lung_mask_root,
        batch["cxr_0"],
        image_size=baseline.shape[-1],
        device=baseline.device,
        dtype=baseline.dtype,
    )


def image_response_metrics(prefix, image, baseline, lung_mask, lower_lung_mask):
    dark = soft_dark_field_mask(image)
    baseline_dark = soft_dark_field_mask(baseline)
    lung_brightness = masked_mean(image, lung_mask)
    baseline_lung_brightness = masked_mean(baseline, lung_mask)
    lower_brightness = masked_mean(image, lower_lung_mask)
    baseline_lower_brightness = masked_mean(baseline, lower_lung_mask)
    lung_dark = masked_mean(dark, lung_mask)
    baseline_lung_dark = masked_mean(baseline_dark, lung_mask)
    lower_dark = masked_mean(dark, lower_lung_mask)
    baseline_lower_dark = masked_mean(baseline_dark, lower_lung_mask)
    return {
        f"{prefix}_lung_brightness": lung_brightness,
        f"{prefix}_lower_lung_brightness": lower_brightness,
        f"{prefix}_lung_dark_fraction": lung_dark,
        f"{prefix}_lower_lung_dark_fraction": lower_dark,
        f"{prefix}_decongestion_brightness": baseline_lung_brightness - lung_brightness,
        f"{prefix}_lower_decongestion_brightness": baseline_lower_brightness - lower_brightness,
        f"{prefix}_dark_response": lung_dark - baseline_lung_dark,
        f"{prefix}_lower_dark_response": lower_dark - baseline_lower_dark,
    }


def treatment_contrast_metrics(pred_a0, pred_a1, baseline, lung_mask, nonlung_mask, lower_lung_mask):
    contrast = (pred_a1 - pred_a0).abs()
    valid = central_field_mask_like(baseline)
    border = border_mask_like(baseline)
    lung_abs = masked_sum(contrast, lung_mask)
    valid_abs = masked_sum(contrast, valid).clamp_min(EPS)
    border_abs = masked_sum(contrast, border)
    nonlung_l1 = masked_l1(pred_a1, pred_a0, nonlung_mask)
    lung_l1 = masked_l1(pred_a1, pred_a0, lung_mask)
    border_l1 = masked_l1(pred_a1, pred_a0, border)
    a0_response = image_response_metrics("a0", pred_a0, baseline, lung_mask, lower_lung_mask)
    a1_response = image_response_metrics("a1", pred_a1, baseline, lung_mask, lower_lung_mask)
    return {
        "swap_l1_global": contrast.mean(dim=(1, 2, 3)),
        "swap_l1_lung": lung_l1,
        "swap_l1_nonlung": nonlung_l1,
        "swap_l1_lower_lung": masked_l1(pred_a1, pred_a0, lower_lung_mask),
        "swap_l1_border": border_l1,
        "swap_lung_fraction": lung_abs / valid_abs,
        "swap_border_fraction": border_abs / valid_abs,
        "swap_localization_ratio": lung_l1 / nonlung_l1.clamp_min(EPS),
        "swap_lung_to_border_ratio": lung_l1 / border_l1.clamp_min(EPS),
        "pred_treatment_effect_decongestion": (
            a1_response["a1_decongestion_brightness"] - a0_response["a0_decongestion_brightness"]
        ),
        "pred_treatment_effect_lower_decongestion": (
            a1_response["a1_lower_decongestion_brightness"]
            - a0_response["a0_lower_decongestion_brightness"]
        ),
        "pred_treatment_effect_dark_response": (
            a1_response["a1_dark_response"] - a0_response["a0_dark_response"]
        ),
        "pred_treatment_effect_lower_dark_response": (
            a1_response["a1_lower_dark_response"] - a0_response["a0_lower_dark_response"]
        ),
    }


def summarize(results):
    summary = {"count": int(len(results))}
    if len(results) == 0:
        return summary
    numeric = results.select_dtypes(include=["number"])
    for column in numeric.columns:
        summary[column] = float(numeric[column].mean())
    if {"target_decongestion_brightness", "factual_decongestion_brightness"}.issubset(results.columns):
        target = pd.to_numeric(results["target_decongestion_brightness"], errors="coerce")
        factual = pd.to_numeric(results["factual_decongestion_brightness"], errors="coerce")
        mask = target.notna() & factual.notna() & (target.abs() > 1e-6) & (factual.abs() > 1e-6)
        summary["factual_decongestion_sign_agreement"] = (
            float((np.sign(target[mask]) == np.sign(factual[mask])).mean()) if int(mask.sum()) else None
        )
        summary["factual_decongestion_mae"] = float((target - factual).abs().mean())
        if int(mask.sum()) >= 3 and float(target[mask].std()) > 0.0 and float(factual[mask].std()) > 0.0:
            summary["factual_decongestion_corr"] = float(np.corrcoef(target[mask], factual[mask])[0, 1])
        else:
            summary["factual_decongestion_corr"] = None
    if "pred_treatment_effect_decongestion" in results.columns:
        treatment_effect = pd.to_numeric(results["pred_treatment_effect_decongestion"], errors="coerce")
        summary["pred_treatment_effect_decongestion_positive_fraction"] = float((treatment_effect > 0.0).mean())
    if "pred_treatment_effect_dark_response" in results.columns:
        dark_effect = pd.to_numeric(results["pred_treatment_effect_dark_response"], errors="coerce")
        summary["pred_treatment_effect_dark_positive_fraction"] = float((dark_effect > 0.0).mean())
    return summary


def subgroup_summary(results):
    rows = [{"group": "overall", "value": "all", **summarize(results)}]
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
    for group_name, values in group_specs:
        temp = results.copy()
        temp[group_name] = values
        for value, group in temp.groupby(group_name, dropna=False):
            rows.append({"group": group_name, "value": str(value), **summarize(group)})
    return pd.DataFrame(rows)


def to_numpy(image):
    return image.detach().float().cpu().numpy()


def save_lung_effect_panel(records, output_path, sample_count):
    selected = records[:sample_count]
    if not selected:
        return
    rows = [
        ("baseline_image", "Baseline", "gray"),
        ("target_image", "Follow-up", "gray"),
        ("pred_a0_image", "Pred a=0", "gray"),
        ("pred_a1_image", "Pred a=1", "gray"),
        ("lung_mask", "Lung mask", "viridis"),
        ("swap_map", "|a=1 - a=0|", "magma"),
    ]
    fig, axes = plt.subplots(len(rows), len(selected), figsize=(3.2 * len(selected), 2.8 * len(rows)))
    if len(selected) == 1:
        axes = axes.reshape(len(rows), 1)
    for col, record in enumerate(selected):
        vmax_swap = max(0.05, float(np.percentile(record["swap_map"], 99)))
        for row_idx, (key, label, cmap) in enumerate(rows):
            axis = axes[row_idx, col]
            value = record[key]
            if key == "swap_map":
                axis.imshow(value, cmap=cmap, vmin=0.0, vmax=vmax_swap)
            elif key == "lung_mask":
                axis.imshow(record["baseline_image"], cmap="gray", vmin=0.0, vmax=1.0)
                axis.imshow(value, cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.35)
            else:
                axis.imshow(value, cmap=cmap, vmin=0.0, vmax=1.0)
                if key in {"pred_a0_image", "pred_a1_image"}:
                    axis.imshow(record["lung_mask"], cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.18)
            if row_idx == 0:
                label = f"a={record['treated_int']}, dt={record['hours_diff']:.1f}h"
            axis.set_title(label)
            axis.axis("off")
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
    loader, split_frame, _ = prepare_eval_data(args, config)
    start_timestep = args.sample_start_timestep or int(config.get("sample_start_timestep", 250))
    sample_steps = args.sample_steps or int(config.get("sample_steps", 50))

    records = []
    image_records = []
    cursor = 0
    for batch in tqdm(loader, desc=f"lung-localized treatment effect {args.split}"):
        batch = move_batch_to_device(batch, device)
        batch_size = batch["x_0"].shape[0]
        metadata = split_frame.iloc[cursor : cursor + batch_size].reset_index(drop=True)
        cursor += batch_size

        batch_a0 = dict(batch)
        batch_a1 = dict(batch)
        batch_a0["a"] = torch.zeros_like(batch["a"])
        batch_a1["a"] = torch.ones_like(batch["a"])

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            paired_noise = torch.randn_like(autoencoder.encode(batch["x_0"], sample=False))
            pred_a0_raw, _ = sample_images(
                autoencoder,
                model,
                mean_head,
                ddpm,
                batch_a0,
                start_timestep,
                sample_steps,
                bridge_bundle=None,
                device=device,
                use_amp=use_amp,
                noise=paired_noise,
            )
            pred_a1_raw, _ = sample_images(
                autoencoder,
                model,
                mean_head,
                ddpm,
                batch_a1,
                start_timestep,
                sample_steps,
                bridge_bundle=None,
                device=device,
                use_amp=use_amp,
                noise=paired_noise,
            )

        baseline = denorm(batch["x_0"])
        target = denorm(batch["y"])
        pred_a0 = denorm(pred_a0_raw)
        pred_a1 = denorm(pred_a1_raw)
        factual = torch.where(batch["a"].view(-1, 1, 1, 1) >= 0.5, pred_a1, pred_a0)
        lung = resolve_lung_mask(args, batch, baseline)
        nonlung = outside_lung_mask(baseline, lung)
        lower_lung = (lower_lung_roi_like(baseline) * lung).clamp(0.0, 1.0)

        target_response = image_response_metrics("target", target, baseline, lung, lower_lung)
        factual_response = image_response_metrics("factual", factual, baseline, lung, lower_lung)
        contrast_metrics = treatment_contrast_metrics(pred_a0, pred_a1, baseline, lung, nonlung, lower_lung)
        factual_lung_mae = masked_l1(factual, target, lung)
        baseline_lung_mae = masked_l1(baseline, target, lung)
        factual_nonlung_mae = masked_l1(factual, target, nonlung)
        factual_lower_lung_mae = masked_l1(factual, target, lower_lung)

        for row in range(batch_size):
            meta = metadata.iloc[row].to_dict()
            record = {
                "example_index": int(len(records)),
                "subject_id": str(batch["subject_id"][row]),
                "cxr_0": str(batch["cxr_0"][row]),
                "cxr_1": str(batch["cxr_1"][row]),
                "treated_int": int(batch["a"][row].item() >= 0.5),
                "hours_diff": float(batch["hours_diff"][row].item()),
                "baseline_lung_mae": float(baseline_lung_mae[row].item()),
                "factual_lung_mae": float(factual_lung_mae[row].item()),
                "factual_nonlung_mae": float(factual_nonlung_mae[row].item()),
                "factual_lower_lung_mae": float(factual_lower_lung_mae[row].item()),
                "lung_mask_mean": float(lung[row].mean().item()),
                "nonlung_mask_mean": float(nonlung[row].mean().item()),
            }
            for key, value in target_response.items():
                record[key] = float(value[row].item())
            for key, value in factual_response.items():
                record[key] = float(value[row].item())
            for key, value in contrast_metrics.items():
                record[key] = float(value[row].item())
            record["factual_decongestion_error"] = (
                record["factual_decongestion_brightness"] - record["target_decongestion_brightness"]
            )
            record["factual_lower_decongestion_error"] = (
                record["factual_lower_decongestion_brightness"]
                - record["target_lower_decongestion_brightness"]
            )
            record["factual_dark_response_error"] = (
                record["factual_dark_response"] - record["target_dark_response"]
            )
            record["factual_lung_mae_improvement_vs_baseline"] = (
                record["baseline_lung_mae"] - record["factual_lung_mae"]
            )
            for key in ("gender", "age_at_t0", "view_0", "view_1", "treatment_status", "balancing_weight"):
                if key in meta:
                    record[key] = meta[key]
            image_record = dict(record)
            image_record.update(
                {
                    "baseline_image": to_numpy(baseline[row, 0]),
                    "target_image": to_numpy(target[row, 0]),
                    "pred_a0_image": to_numpy(pred_a0[row, 0]),
                    "pred_a1_image": to_numpy(pred_a1[row, 0]),
                    "lung_mask": to_numpy(lung[row, 0]),
                    "swap_map": to_numpy((pred_a1[row, 0] - pred_a0[row, 0]).abs()),
                }
            )
            records.append(record)
            image_records.append(image_record)

    results = pd.DataFrame(records)
    results_path = output_dir / "lung_localized_treatment_effect_examples.csv"
    summary_path = output_dir / "lung_localized_treatment_effect_summary.json"
    subgroup_path = output_dir / "lung_localized_treatment_effect_subgroups.csv"
    results.to_csv(results_path, index=False)
    subgroup_summary(results).to_csv(subgroup_path, index=False)

    summary = summarize(results)
    summary.update(
        {
            "split": args.split,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "target_mode": config.get("target_mode", "target"),
            "mean_architecture": config.get("mean_architecture", "standard"),
            "sample_start_timestep": start_timestep,
            "sample_steps": sample_steps,
            "mask_method": args.mask_method,
            "lung_mask_root": args.lung_mask_root or None,
            "seed": args.seed,
            "notes": [
                (
                    "The lung mask is loaded from a cached anatomical segmentation."
                    if args.mask_method == "precomputed"
                    else "The lung mask is a label-free proxy, not a validated anatomical segmentation."
                ),
                "Positive pred_treatment_effect_decongestion means predicted a=1 is darker in the lung mask than predicted a=0.",
                "Positive pred_treatment_effect_dark_response means predicted a=1 has larger dark-field fraction than predicted a=0.",
            ],
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    best_localized = sorted(
        image_records,
        key=lambda item: (
            item["swap_lung_fraction"],
            item["swap_lung_to_border_ratio"],
            -item["factual_lung_mae"],
        ),
        reverse=True,
    )[: args.sample_count]
    largest_effect = sorted(image_records, key=lambda item: item["swap_l1_lung"], reverse=True)[
        : args.sample_count
    ]
    random_panel = image_records[: args.sample_count]
    save_lung_effect_panel(best_localized, output_dir / "best_lung_localized_effect_panel.png", args.sample_count)
    save_lung_effect_panel(largest_effect, output_dir / "largest_lung_effect_panel.png", args.sample_count)
    save_lung_effect_panel(random_panel, output_dir / "random_lung_effect_panel.png", args.sample_count)

    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
