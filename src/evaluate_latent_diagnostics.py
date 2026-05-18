#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CXRPairDataset, build_transform
from evaluate_latent_diffusion import (
    compute_artifacts,
    compute_ssim,
    create_ssim_window,
    denorm,
    load_latent_checkpoint,
    prepare_eval_data,
    sample_images,
    soft_dark_field_mask,
    soft_dice,
    total_variation,
)
from experiment_utils import move_batch_to_device
from experiment_utils import apply_conditioning_mode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose mean-only, residual diffusion, and autoencoder ceilings."
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
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--sample-start-timestep", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--skip-image-verify", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_tensors(prefix, prediction, target, baseline, target_mask, ssim_window):
    prediction_mask = soft_dark_field_mask(prediction)
    artifacts = compute_artifacts(target, prediction, target_mask, prediction_mask)
    target_change = (target - baseline).abs().mean(dim=(1, 2, 3))
    prediction_change = (prediction - baseline).abs().mean(dim=(1, 2, 3))
    return {
        f"{prefix}_mae": (prediction - target).abs().mean(dim=(1, 2, 3)),
        f"{prefix}_ssim": compute_ssim(prediction, target, ssim_window),
        f"{prefix}_dark_field_dice": soft_dice(prediction_mask, target_mask),
        f"{prefix}_lower_saturation_excess": artifacts["lower_saturation_excess"],
        f"{prefix}_border_abs_diff": artifacts["border_abs_diff"],
        f"{prefix}_high_frequency_ratio": artifacts["high_frequency_ratio"],
        f"{prefix}_tv_ratio": total_variation(prediction) / total_variation(target).clamp_min(1e-8),
        f"{prefix}_change_ratio": prediction_change / target_change.clamp_min(1e-8),
        f"{prefix}_mean_intensity": prediction.mean(dim=(1, 2, 3)),
    }


def summarize(results):
    summary = {"count": int(len(results))}
    metric_columns = [
        column
        for column in results.columns
        if column.endswith(
            (
                "_mae",
                "_ssim",
                "_dark_field_dice",
                "_lower_saturation_excess",
                "_border_abs_diff",
                "_high_frequency_ratio",
                "_tv_ratio",
                "_change_ratio",
                "_mean_intensity",
                "_l1",
            )
        )
    ]
    for column in metric_columns:
        summary[column] = float(results[column].mean()) if len(results) else None

    if len(results) and "mean_mae" in results.columns and "full_mae" in results.columns:
        summary["full_minus_mean_mae"] = float((results["full_mae"] - results["mean_mae"]).mean())
        summary["full_minus_mean_ssim"] = float((results["full_ssim"] - results["mean_ssim"]).mean())
        summary["full_minus_mean_dice"] = float(
            (results["full_dark_field_dice"] - results["mean_dark_field_dice"]).mean()
        )
        summary["full_minus_mean_change_ratio"] = float(
            (results["full_change_ratio"] - results["mean_change_ratio"]).mean()
        )
        summary["mean_improvement_vs_baseline_mae"] = float(
            (results["baseline_mae"] - results["mean_mae"]).mean()
        )
        summary["full_improvement_vs_baseline_mae"] = float(
            (results["baseline_mae"] - results["full_mae"]).mean()
        )
        summary["full_improvement_vs_mean_mae"] = float(
            (results["mean_mae"] - results["full_mae"]).mean()
        )
        summary["full_improvement_vs_mean_ssim"] = float(
            (results["full_ssim"] - results["mean_ssim"]).mean()
        )
    if len(results) and "ae_mae" in results.columns:
        summary["mean_gap_to_ae_mae"] = (
            float((results["mean_mae"] - results["ae_mae"]).mean())
            if "mean_mae" in results.columns
            else None
        )
        summary["full_gap_to_ae_mae"] = (
            float((results["full_mae"] - results["ae_mae"]).mean())
            if "full_mae" in results.columns
            else None
        )
    return summary


def subgroup_summary(results):
    rows = []
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
    for group_name, group_values in group_specs:
        temp = results.copy()
        temp[group_name] = group_values
        for value, group in temp.groupby(group_name, dropna=False):
            row = {"group": group_name, "value": str(value)}
            row.update(summarize(group))
            rows.append(row)
    return pd.DataFrame(rows)


def save_diagnostic_panel(records, output_path, sample_count):
    selected = records[:sample_count]
    if not selected:
        return
    rows = [
        ("baseline_image", "Baseline"),
        ("target_image", "Follow-up"),
        ("ae_image", "AE follow-up recon"),
        ("mean_image", "Mean-only"),
        ("full_image", "Mean + residual DDPM"),
        ("full_abs_error_image", "Full abs error"),
    ]
    fig, axes = plt.subplots(len(rows), len(selected), figsize=(4 * len(selected), 3 * len(rows)))
    if len(selected) == 1:
        axes = axes.reshape(len(rows), 1)
    for col, record in enumerate(selected):
        for row_idx, (key, label) in enumerate(rows):
            axes[row_idx, col].imshow(record[key], cmap="gray", vmin=0.0, vmax=1.0)
            if row_idx == 0:
                label = f"a={record['treated_int']}, dt={record['hours_diff']:.1f}h"
            axes[row_idx, col].set_title(label)
            axes[row_idx, col].axis("off")
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
    checkpoint, config, autoencoder, model, ddpm, mean_head = load_latent_checkpoint(
        args.checkpoint,
        device,
    )
    if config.get("target_mode", "target") != "mean_residual" or mean_head is None:
        raise ValueError("This diagnostic currently requires a mean_residual checkpoint.")

    loader, split_frame, image_size = prepare_eval_data(args, config)
    start_timestep = args.sample_start_timestep or int(config.get("sample_start_timestep", 250))
    sample_steps = args.sample_steps or int(config.get("sample_steps", 50))
    treatment_conditioning_mode = config.get("treatment_conditioning_mode", "full")
    ssim_window = create_ssim_window(11, 1.5, device)

    records = []
    image_records = []
    cursor = 0
    for batch in tqdm(loader, desc=f"diagnose latent {args.split}"):
        batch = move_batch_to_device(batch, device)
        batch_size = batch["x_0"].shape[0]
        metadata = split_frame.iloc[cursor : cursor + batch_size].reset_index(drop=True)
        cursor += batch_size

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            baseline_latent = autoencoder.encode(batch["x_0"], sample=False)
            target_latent = autoencoder.encode(batch["y"], sample=False)
            ae_raw = autoencoder.decode(target_latent)
            treatment = apply_conditioning_mode(batch["a"], treatment_conditioning_mode)
            if getattr(mean_head, "supports_potential_outcomes", False):
                mean_latent, mu0_latent, tau_latent = mean_head(
                    baseline_latent,
                    treatment,
                    batch["delta"],
                    return_components=True,
                )
            else:
                mean_latent = mean_head(baseline_latent, treatment, batch["delta"])
                mu0_latent = None
                tau_latent = None
            mean_raw = autoencoder.decode(mean_latent)
            paired_noise = torch.randn_like(baseline_latent)
            full_raw, _ = sample_images(
                autoencoder,
                model,
                mean_head,
                ddpm,
                batch,
                start_timestep,
                sample_steps,
                bridge_bundle=None,
                device=device,
                use_amp=use_amp,
                noise=paired_noise,
            )
            swapped_batch = dict(batch)
            swapped_batch["a"] = 1.0 - batch["a"]
            swapped_treatment = apply_conditioning_mode(
                swapped_batch["a"],
                treatment_conditioning_mode,
            )
            swapped_mean_latent = mean_head(baseline_latent, swapped_treatment, batch["delta"])
            swapped_mean_raw = autoencoder.decode(swapped_mean_latent)
            swapped_full_raw, _ = sample_images(
                autoencoder,
                model,
                mean_head,
                ddpm,
                swapped_batch,
                start_timestep,
                sample_steps,
                bridge_bundle=None,
                device=device,
                use_amp=use_amp,
                noise=paired_noise,
            )

        baseline = denorm(batch["x_0"])
        target = denorm(batch["y"])
        ae_recon = denorm(ae_raw)
        mean_prediction = denorm(mean_raw)
        full_prediction = denorm(full_raw)
        swapped_mean = denorm(swapped_mean_raw)
        swapped_full = denorm(swapped_full_raw)
        target_mask = soft_dark_field_mask(target)

        all_metrics = {}
        for prefix, prediction in (
            ("baseline", baseline),
            ("ae", ae_recon),
            ("mean", mean_prediction),
            ("full", full_prediction),
        ):
            all_metrics.update(
                metric_tensors(
                    prefix,
                    prediction,
                    target,
                    baseline,
                    target_mask,
                    ssim_window,
                )
            )
        all_metrics["full_vs_mean_l1"] = (full_prediction - mean_prediction).abs().mean(dim=(1, 2, 3))
        all_metrics["mean_swap_l1"] = (swapped_mean - mean_prediction).abs().mean(dim=(1, 2, 3))
        all_metrics["full_swap_l1"] = (swapped_full - full_prediction).abs().mean(dim=(1, 2, 3))
        if tau_latent is not None:
            all_metrics["tau_latent_l1"] = tau_latent.abs().mean(dim=(1, 2, 3))
            all_metrics["factual_tau_latent_l1"] = (
                treatment.view(-1, 1, 1, 1) * tau_latent
            ).abs().mean(dim=(1, 2, 3))
            all_metrics["mu0_vs_mean_l1"] = (mean_latent - mu0_latent).abs().mean(dim=(1, 2, 3))

        for row in range(batch_size):
            meta = metadata.iloc[row].to_dict()
            record = {
                "example_index": int(len(records)),
                "subject_id": str(batch["subject_id"][row]),
                "cxr_0": str(batch["cxr_0"][row]),
                "cxr_1": str(batch["cxr_1"][row]),
                "treated_int": int(batch["a"][row].item() >= 0.5),
                "hours_diff": float(batch["hours_diff"][row].item()),
            }
            for key, value in all_metrics.items():
                record[key] = float(value[row].item())
            for key in ("gender", "age_at_t0", "view_0", "view_1", "treatment_status", "balancing_weight"):
                if key in meta:
                    record[key] = meta[key]
            image_record = dict(record)
            image_record.update(
                {
                    "baseline_image": baseline[row, 0].detach().cpu().numpy(),
                    "target_image": target[row, 0].detach().cpu().numpy(),
                    "ae_image": ae_recon[row, 0].detach().cpu().numpy(),
                    "mean_image": mean_prediction[row, 0].detach().cpu().numpy(),
                    "full_image": full_prediction[row, 0].detach().cpu().numpy(),
                    "full_abs_error_image": (
                        full_prediction[row, 0] - target[row, 0]
                    ).abs().detach().cpu().numpy(),
                }
            )
            records.append(record)
            image_records.append(image_record)

    results = pd.DataFrame(records)
    results.to_csv(output_dir / "latent_diagnostics_examples.csv", index=False)
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
            "treatment_conditioning_mode": treatment_conditioning_mode,
            "seed": args.seed,
        }
    )
    (output_dir / "latent_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    subgroup_summary(results).to_csv(output_dir / "latent_diagnostics_subgroups.csv", index=False)

    random_records = image_records[: args.sample_count]
    best_full = sorted(image_records, key=lambda item: item["full_mae"])[: args.sample_count]
    largest_full_gain = sorted(
        image_records,
        key=lambda item: item["mean_mae"] - item["full_mae"],
        reverse=True,
    )[: args.sample_count]
    save_diagnostic_panel(random_records, output_dir / "diagnostic_random_panel.png", args.sample_count)
    save_diagnostic_panel(best_full, output_dir / "diagnostic_best_full_panel.png", args.sample_count)
    save_diagnostic_panel(
        largest_full_gain,
        output_dir / "diagnostic_largest_residual_gain_panel.png",
        args.sample_count,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
