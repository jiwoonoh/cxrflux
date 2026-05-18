import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CXRPairDataset, build_transform, prepare_pair_splits
from ddpm import DDPM
from experiment_utils import (
    apply_conditioning_mode,
    make_fixed_noise,
    move_batch_to_device,
    set_seed,
)
from train import create_model, evaluate, save_comparison_panel, save_treatment_swap_panel


TIME_GAP_BINS = [0.0, 12.0, 24.0, 36.0, 48.0]
TIME_GAP_LABELS = ["0-12h", "12-24h", "24-36h", "36-48h"]
SUMMARY_COLUMNS = [
    "mae",
    "mse",
    "psnr",
    "ssim",
    "swap_l1",
    "factual_change_l1",
    "swap_to_change_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate held-out conditional CXR generation")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--sample-seed", type=int, default=4242)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--beta-schedule", choices=["linear", "cosine"], default=None)
    parser.add_argument("--conditioning-mode", choices=["full", "no_treatment"], default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--evaluation-sampler",
        choices=["deterministic", "stochastic"],
        default="deterministic",
    )
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def resolve_config(args, checkpoint_config):
    checkpoint_path = Path(args.checkpoint)
    csv_path = args.csv_path or checkpoint_config.get("csv_path", "cxr_pairs_frontal.csv")
    image_root = args.image_root or checkpoint_config.get("image_root", "./mimic-cxr-jpg/")
    seed = args.seed if args.seed is not None else checkpoint_config.get("seed", 42)
    image_size = args.image_size or checkpoint_config.get("image_size", 128)
    base_channels = args.base_channels or checkpoint_config.get("base_channels", 64)
    num_timesteps = args.num_timesteps or checkpoint_config.get("num_timesteps", 1000)
    beta_schedule = args.beta_schedule or checkpoint_config.get("beta_schedule", "cosine")
    conditioning_mode = args.conditioning_mode or checkpoint_config.get(
        "conditioning_mode", "full"
    )
    split_dir = args.split_dir or checkpoint_config.get(
        "split_dir",
        str(Path("splits") / f"{Path(csv_path).stem}_seed{seed}"),
    )
    output_dir = args.output_dir or str(Path("metrics") / checkpoint_path.stem)

    return {
        "checkpoint_path": str(checkpoint_path),
        "csv_path": csv_path,
        "image_root": image_root,
        "seed": seed,
        "eval_seed": args.eval_seed,
        "sample_seed": args.sample_seed,
        "image_size": image_size,
        "base_channels": base_channels,
        "num_timesteps": num_timesteps,
        "beta_schedule": beta_schedule,
        "conditioning_mode": conditioning_mode,
        "split_dir": split_dir,
        "output_dir": output_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "sample_count": args.sample_count,
        "max_examples": args.max_examples,
        "use_amp": not args.disable_amp,
        "evaluation_sampler": args.evaluation_sampler,
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

    mu_prediction_sq = mu_prediction.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_prediction_target = mu_prediction * mu_target

    sigma_prediction_sq = F.conv2d(prediction * prediction, window, padding=padding) - mu_prediction_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=padding) - mu_target_sq
    sigma_prediction_target = (
        F.conv2d(prediction * target, window, padding=padding) - mu_prediction_target
    )

    ssim_map = (
        (2 * mu_prediction_target + c1)
        * (2 * sigma_prediction_target + c2)
        / ((mu_prediction_sq + mu_target_sq + c1) * (sigma_prediction_sq + sigma_target_sq + c2))
    )
    return ssim_map.mean(dim=(1, 2, 3))


def compute_total_variation_score(image):
    diff_x = image[..., 1:] - image[..., :-1]
    diff_y = image[:, :, 1:, :] - image[:, :, :-1, :]
    return diff_x.abs().mean(dim=(1, 2, 3)) + diff_y.abs().mean(dim=(1, 2, 3))


def generate_factual_and_swapped(
    model,
    ddpm,
    device_batch,
    image_size,
    noise,
    conditioning_mode,
    deterministic_sampling,
):
    requested_treatment = device_batch["a"]
    treatment_zero = apply_conditioning_mode(torch.zeros_like(requested_treatment), conditioning_mode)
    treatment_one = apply_conditioning_mode(torch.ones_like(requested_treatment), conditioning_mode)

    factual_zero = ddpm.sample(
        model,
        device_batch["x_0"],
        treatment_zero,
        device_batch["delta"],
        image_size=image_size,
        initial_noise=noise,
        stochastic=not deterministic_sampling,
    )
    factual_one = ddpm.sample(
        model,
        device_batch["x_0"],
        treatment_one,
        device_batch["delta"],
        image_size=image_size,
        initial_noise=noise,
        stochastic=not deterministic_sampling,
    )

    treatment_mask = requested_treatment.view(-1, 1, 1, 1) >= 0.5
    factual = torch.where(treatment_mask, factual_one, factual_zero)
    return factual, factual_zero, factual_one


def build_batch_from_indices(pairs_df, image_root, image_size, indices):
    if not indices:
        raise ValueError("Need at least one example index to build a presentation batch")

    batch_frame = pairs_df.iloc[indices].copy().reset_index(drop=True)
    dataset = CXRPairDataset(batch_frame, image_root, transform=build_transform(image_size))
    loader = DataLoader(dataset, batch_size=len(batch_frame), shuffle=False, num_workers=0)
    return next(iter(loader)), batch_frame


def summarize_frame(frame):
    summary = {"count": int(len(frame))}
    if len(frame) == 0:
        for column in SUMMARY_COLUMNS:
            summary[column] = None
        return summary

    for column in SUMMARY_COLUMNS:
        summary[column] = float(frame[column].mean())
    return summary


def build_summary(results_df, test_loss):
    summary = {
        "test_loss": float(test_loss),
        "evaluation_sampler": results_df["evaluation_sampler"].iloc[0] if not results_df.empty else None,
        "overall": summarize_frame(results_df),
        "by_treatment": {},
        "by_time_gap": {},
        "by_treatment_and_time_gap": {},
    }

    for treated_value, group in results_df.groupby("treated_int"):
        summary["by_treatment"][str(int(treated_value))] = summarize_frame(group)

    for time_gap_bin, group in results_df.groupby("time_gap_bin", dropna=False):
        summary["by_time_gap"][str(time_gap_bin)] = summarize_frame(group)

    combined = results_df.groupby(["treated_int", "time_gap_bin"], dropna=False)
    for (treated_value, time_gap_bin), group in combined:
        key = f"treated_{int(treated_value)}__{time_gap_bin}"
        summary["by_treatment_and_time_gap"][key] = summarize_frame(group)

    return summary


def select_presentation_indices(results_df, sample_count):
    if results_df.empty:
        return []

    candidate_frame = results_df.copy()
    artifact_low = candidate_frame["target_artifact_score"].quantile(0.10)
    artifact_high = candidate_frame["target_artifact_score"].quantile(0.90)
    mean_low = candidate_frame["target_mean_intensity"].quantile(0.10)
    mean_high = candidate_frame["target_mean_intensity"].quantile(0.90)
    generated_artifact_low = candidate_frame["generated_artifact_score"].quantile(0.05)
    generated_artifact_high = candidate_frame["generated_artifact_score"].quantile(0.95)
    generated_mean_low = candidate_frame["generated_mean_intensity"].quantile(0.05)
    generated_mean_high = candidate_frame["generated_mean_intensity"].quantile(0.95)
    mae_high = candidate_frame["mae"].quantile(0.90)

    candidate_frame = candidate_frame[
        candidate_frame["target_artifact_score"].between(artifact_low, artifact_high)
        & candidate_frame["target_mean_intensity"].between(mean_low, mean_high)
        & candidate_frame["generated_artifact_score"].between(
            generated_artifact_low, generated_artifact_high
        )
        & candidate_frame["generated_mean_intensity"].between(
            generated_mean_low, generated_mean_high
        )
        & candidate_frame["mae"].le(mae_high)
    ].copy()
    if candidate_frame.empty:
        candidate_frame = results_df.copy()

    for column in [
        "baseline_artifact_score",
        "target_artifact_score",
        "generated_artifact_score",
        "baseline_mean_intensity",
        "target_mean_intensity",
        "generated_mean_intensity",
    ]:
        median = candidate_frame[column].median()
        scale = max((candidate_frame[column] - median).abs().median(), 1e-6)
        candidate_frame[f"{column}_distance"] = (
            candidate_frame[column] - median
        ).abs() / scale

    candidate_frame["presentation_score"] = (
        candidate_frame["baseline_artifact_score_distance"]
        + candidate_frame["target_artifact_score_distance"]
        + candidate_frame["generated_artifact_score_distance"]
        + candidate_frame["baseline_mean_intensity_distance"]
        + candidate_frame["target_mean_intensity_distance"]
        + candidate_frame["generated_mean_intensity_distance"]
        + candidate_frame["mae"].rank(method="average", pct=True)
    )
    candidate_frame = candidate_frame.sort_values(
        ["presentation_score", "hours_diff", "example_index"]
    )

    desired_per_treatment = {
        0: sample_count // 2,
        1: sample_count - (sample_count // 2),
    }

    selected = []
    for treated_value in (0, 1):
        pool = candidate_frame[candidate_frame["treated_int"] == treated_value]
        selected.extend(pool.head(desired_per_treatment[treated_value])["example_index"].tolist())

    selected = list(dict.fromkeys(selected))
    if len(selected) < sample_count:
        remaining = candidate_frame[
            ~candidate_frame["example_index"].isin(selected)
        ]["example_index"].tolist()
        selected.extend(remaining[: sample_count - len(selected)])

    return selected[:sample_count]


def print_summary(summary):
    overall = summary["overall"]
    print("\n=== Held-out Test Metrics ===")
    print(f"Test diffusion loss: {summary['test_loss']:.4f}")
    print(f"Evaluation sampler: {summary['evaluation_sampler']}")
    for metric_name in SUMMARY_COLUMNS:
        print(f"{metric_name}: {overall[metric_name]:.4f}")

    print("\n=== By Treatment ===")
    for treated_key, group_summary in summary["by_treatment"].items():
        print(
            f"treated={treated_key}: n={group_summary['count']}, "
            f"mae={group_summary['mae']:.4f}, ssim={group_summary['ssim']:.4f}, "
            f"swap_ratio={group_summary['swap_to_change_ratio']:.4f}"
        )

    print("\n=== By Time Gap ===")
    for time_gap_key, group_summary in summary["by_time_gap"].items():
        if group_summary["count"] == 0:
            continue
        print(
            f"{time_gap_key}: n={group_summary['count']}, "
            f"mae={group_summary['mae']:.4f}, ssim={group_summary['ssim']:.4f}, "
            f"swap_ratio={group_summary['swap_to_change_ratio']:.4f}"
        )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    config = resolve_config(args, checkpoint_config)
    use_amp = config["use_amp"] and device.type == "cuda"
    deterministic_sampling = config["evaluation_sampler"] == "deterministic"

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config["eval_seed"])

    split_frames = prepare_pair_splits(
        csv_path=config["csv_path"],
        image_root=config["image_root"],
        split_dir=config["split_dir"],
        seed=config["seed"],
        split_fractions=(
            checkpoint_config.get("train_fraction", 0.8),
            checkpoint_config.get("val_fraction", 0.1),
            checkpoint_config.get("test_fraction", 0.1),
        ),
    )

    eval_frame = split_frames["test"].copy().reset_index(drop=True)
    if config["max_examples"] is not None:
        eval_frame = eval_frame.iloc[: config["max_examples"]].copy().reset_index(drop=True)
        print(f"[metrics] Limiting evaluation to the first {len(eval_frame)} held-out examples")
    if eval_frame.empty:
        raise ValueError("The evaluation split is empty; increase --max-examples or check the split")

    transform = build_transform(config["image_size"])
    dataset = CXRPairDataset(eval_frame, config["image_root"], transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config["num_workers"] > 0,
    )

    model = create_model(config, device)
    state_dict = checkpoint.get("ema_model_state_dict") or checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    ddpm = DDPM(
        num_timesteps=config["num_timesteps"],
        beta_schedule=config["beta_schedule"],
        device=device,
    )

    set_seed(config["eval_seed"])
    test_loss = evaluate(
        model,
        ddpm,
        loader,
        device,
        use_amp,
        config["conditioning_mode"],
    )

    print(
        f"[metrics] Evaluating {len(eval_frame)} held-out pairs "
        f"with conditioning_mode={config['conditioning_mode']} "
        f"and evaluation_sampler={config['evaluation_sampler']}"
    )

    ssim_window = create_ssim_window(window_size=11, sigma=1.5, device=device)
    results = []
    example_index = 0

    for batch_index, batch in enumerate(tqdm(loader, desc="Generating held-out predictions")):
        device_batch = move_batch_to_device(batch, device)
        noise = make_fixed_noise(
            device_batch["x_0"].shape[0],
            config["image_size"],
            device,
            config["sample_seed"] + batch_index,
        )

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                factual, factual_zero, factual_one = generate_factual_and_swapped(
                    model,
                    ddpm,
                    device_batch,
                    config["image_size"],
                    noise,
                    config["conditioning_mode"],
                    deterministic_sampling,
                )

        baseline = device_batch["x_0"].detach().float().clamp(-1.0, 1.0).add(1.0).div(2.0)
        target = device_batch["y"].detach().float().clamp(-1.0, 1.0).add(1.0).div(2.0)
        factual = factual.detach().float().clamp(-1.0, 1.0).add(1.0).div(2.0)
        factual_zero = factual_zero.detach().float().clamp(-1.0, 1.0).add(1.0).div(2.0)
        factual_one = factual_one.detach().float().clamp(-1.0, 1.0).add(1.0).div(2.0)

        mae = (factual - target).abs().mean(dim=(1, 2, 3))
        mse = (factual - target).pow(2).mean(dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
        ssim = compute_ssim(factual, target, ssim_window)
        swap_l1 = (factual_one - factual_zero).abs().mean(dim=(1, 2, 3))
        factual_change_l1 = (factual - baseline).abs().mean(dim=(1, 2, 3))
        swap_to_change_ratio = swap_l1 / factual_change_l1.clamp_min(1e-8)
        baseline_artifact_score = compute_total_variation_score(baseline)
        target_artifact_score = compute_total_variation_score(target)
        generated_artifact_score = compute_total_variation_score(factual)
        baseline_mean_intensity = baseline.mean(dim=(1, 2, 3))
        target_mean_intensity = target.mean(dim=(1, 2, 3))
        generated_mean_intensity = factual.mean(dim=(1, 2, 3))

        batch_size = factual.shape[0]
        for row_offset in range(batch_size):
            results.append(
                {
                    "example_index": example_index + row_offset,
                    "subject_id": batch["subject_id"][row_offset],
                    "cxr_0": batch["cxr_0"][row_offset],
                    "cxr_1": batch["cxr_1"][row_offset],
                    "treated": float(batch["a"][row_offset].item()),
                    "treated_int": int(batch["a"][row_offset].item()),
                    "hours_diff": float(batch["hours_diff"][row_offset].item()),
                    "mae": float(mae[row_offset].item()),
                    "mse": float(mse[row_offset].item()),
                    "psnr": float(psnr[row_offset].item()),
                    "ssim": float(ssim[row_offset].item()),
                    "swap_l1": float(swap_l1[row_offset].item()),
                    "factual_change_l1": float(factual_change_l1[row_offset].item()),
                    "swap_to_change_ratio": float(swap_to_change_ratio[row_offset].item()),
                    "baseline_artifact_score": float(baseline_artifact_score[row_offset].item()),
                    "target_artifact_score": float(target_artifact_score[row_offset].item()),
                    "generated_artifact_score": float(generated_artifact_score[row_offset].item()),
                    "baseline_mean_intensity": float(baseline_mean_intensity[row_offset].item()),
                    "target_mean_intensity": float(target_mean_intensity[row_offset].item()),
                    "generated_mean_intensity": float(generated_mean_intensity[row_offset].item()),
                    "evaluation_sampler": config["evaluation_sampler"],
                }
            )
        example_index += batch_size

    results_df = pd.DataFrame(results)
    results_df["time_gap_bin"] = pd.cut(
        results_df["hours_diff"],
        bins=TIME_GAP_BINS,
        labels=TIME_GAP_LABELS,
        include_lowest=True,
    ).astype(str)

    summary = build_summary(results_df, test_loss)

    summary_path = output_dir / "metrics_summary.json"
    per_example_path = output_dir / "metrics_per_example.csv"
    presentation_path = output_dir / "presentation_examples.csv"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    results_df.to_csv(per_example_path, index=False)

    selected_indices = select_presentation_indices(results_df, config["sample_count"])
    selected_batch, _ = build_batch_from_indices(
        eval_frame,
        config["image_root"],
        config["image_size"],
        selected_indices,
    )
    selected_metadata = (
        results_df.set_index("example_index").loc[selected_indices].reset_index()
    )
    selected_metadata.to_csv(presentation_path, index=False)

    panel_noise = make_fixed_noise(
        len(selected_indices),
        config["image_size"],
        device,
        config["sample_seed"] + 10_000,
    )
    save_comparison_panel(
        model,
        ddpm,
        selected_batch,
        device,
        config["image_size"],
        panel_noise,
        output_dir / "paper_test_samples.png",
        config["conditioning_mode"],
        deterministic_sampling=deterministic_sampling,
    )
    save_treatment_swap_panel(
        model,
        ddpm,
        selected_batch,
        device,
        config["image_size"],
        panel_noise,
        output_dir / "paper_test_treatment_swap.png",
        config["conditioning_mode"],
        deterministic_sampling=True,
    )

    print_summary(summary)
    print(f"\nSaved summary JSON to {summary_path}")
    print(f"Saved per-example CSV to {per_example_path}")
    print(f"Saved presentation metadata to {presentation_path}")
    print(f"Saved paper-ready panels to {output_dir}")


if __name__ == "__main__":
    main()
