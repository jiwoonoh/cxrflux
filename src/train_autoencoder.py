#!/usr/bin/env python3

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from autoencoder import CXRLatentAutoencoder, kl_divergence
from dataset import build_transform


ImageFile.LOAD_TRUNCATED_IMAGES = True


FINAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = FINAL_ROOT / "results" / "runs" / "latent_diffusion_cxr256_v1"


class CXRImageDataset(Dataset):
    def __init__(
        self,
        manifest_csv,
        image_size,
        verify_readable=True,
        foreground_crop=False,
        crop_threshold=10,
        crop_min_content_fraction=0.02,
        crop_margin_fraction=0.03,
    ):
        self.frame = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.manifest_root = Path(manifest_csv).resolve().parent
        self.transform = build_transform(
            image_size,
            foreground_crop=foreground_crop,
            crop_threshold=crop_threshold,
            crop_min_content_fraction=crop_min_content_fraction,
            crop_margin_fraction=crop_margin_fraction,
        )
        if verify_readable:
            readable = self.frame["path"].map(self._is_readable)
            dropped = int((~readable).sum())
            if dropped:
                print(f"[data] Dropped {dropped} unreadable images from {manifest_csv}")
            self.frame = self.frame[readable].reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self._resolve_path(row["path"])) as image:
            image = image.convert("L")
            image.load()
        return {
            "image": self.transform(image),
            "path": row["path"],
            "subject_id": str(row["subject_id"]),
            "dicom_id": str(row["dicom_id"]),
        }

    @staticmethod
    def _path_candidates(path, manifest_root):
        path = Path(str(path))
        if path.is_absolute():
            return [path]
        final_root = Path(__file__).resolve().parents[1]
        return [
            path,
            final_root / path,
            final_root / "data" / "mimic-cxr-jpg" / path.name,
        ]

    def _resolve_path(self, path):
        candidates = self._path_candidates(path, self.manifest_root)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]

    def _is_readable(self, path):
        try:
            for candidate in self._path_candidates(path, self.manifest_root):
                if candidate.exists():
                    with Image.open(candidate) as image:
                        image.load()
                    return True
            return False
        except (OSError, IOError):
            return False


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
    parser = argparse.ArgumentParser(description="Train a CXR latent autoencoder.")
    parser.add_argument("--manifest-dir", default=str(FINAL_ROOT / "results" / "manifests" / "autoencoder"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "autoencoder_cxr256_v1"))
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--foreground-crop", action="store_true")
    parser.add_argument("--crop-threshold", type=int, default=10)
    parser.add_argument("--crop-min-content-fraction", type=float, default=0.02)
    parser.add_argument("--crop-margin-fraction", type=float, default=0.03)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=0.25)
    parser.add_argument("--gradient-weight", type=float, default=0.15)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--saturation-weight", type=float, default=0.10)
    parser.add_argument("--border-weight", type=float, default=0.05)
    parser.add_argument("--lower-band-weight", type=float, default=0.05)
    parser.add_argument("--sample-count", type=int, default=8)
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


def gradient_loss(prediction, target):
    dx_pred = prediction[..., 1:] - prediction[..., :-1]
    dx_target = target[..., 1:] - target[..., :-1]
    dy_pred = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (dx_pred - dx_target).abs().mean() + (dy_pred - dy_target).abs().mean()


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


def soft_dice(mask_a, mask_b):
    numerator = 2.0 * (mask_a * mask_b).sum(dim=(1, 2, 3))
    denominator = mask_a.sum(dim=(1, 2, 3)) + mask_b.sum(dim=(1, 2, 3))
    return numerator / denominator.clamp_min(1e-8)


def artifact_penalty(reconstruction, target):
    rec = denorm(reconstruction)
    tgt = denorm(target)
    saturation = ((rec < 0.01).float() + (rec > 0.99).float()).mean()
    border = border_mask_like(rec)
    border_diff = ((rec - tgt).abs() * border).sum() / border.sum().clamp_min(1.0)
    lower_start = int(round(0.78 * rec.shape[-2]))
    lower_diff = (rec[..., lower_start:, :] - tgt[..., lower_start:, :]).abs().mean()
    return saturation, border_diff, lower_diff


def autoencoder_loss(model, batch, ssim_window, args):
    image = batch["image"]
    reconstruction, mu, logvar = model(image, sample=True)
    image_01 = denorm(image)
    reconstruction_01 = denorm(reconstruction)
    l1 = (reconstruction - image).abs().mean()
    ssim = compute_ssim(reconstruction_01, image_01, ssim_window).mean()
    grad = gradient_loss(reconstruction, image)
    kl = kl_divergence(mu, logvar)
    saturation, border, lower = artifact_penalty(reconstruction, image)
    loss = (
        args.l1_weight * l1
        + args.ssim_weight * (1.0 - ssim)
        + args.gradient_weight * grad
        + args.kl_weight * kl
        + args.saturation_weight * saturation
        + args.border_weight * border
        + args.lower_band_weight * lower
    )
    metrics = {
        "loss": loss.detach(),
        "l1": l1.detach(),
        "ssim": ssim.detach(),
        "gradient": grad.detach(),
        "kl": kl.detach(),
        "saturation": saturation.detach(),
        "border": border.detach(),
        "lower": lower.detach(),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    ssim_window = create_ssim_window(11, 1.5, device)
    totals = {
        "mae": 0.0,
        "ssim": 0.0,
        "dark_field_dice": 0.0,
        "lower_saturation_excess": 0.0,
    }
    count = 0
    random_records = []
    worst_records = []
    for batch in tqdm(loader, desc="autoencoder eval"):
        image = batch["image"].to(device)
        reconstruction, _, _ = model(image, sample=False)
        image_01 = denorm(image)
        reconstruction_01 = denorm(reconstruction)
        mae = (reconstruction_01 - image_01).abs().mean(dim=(1, 2, 3))
        ssim = compute_ssim(reconstruction_01, image_01, ssim_window)
        dark_dice = soft_dice(
            soft_dark_field_mask(reconstruction_01),
            soft_dark_field_mask(image_01),
        )
        lower_start = int(round(0.78 * image_01.shape[-2]))
        lower_sat = (
            (reconstruction_01[..., lower_start:, :] > 0.96).float().mean(dim=(1, 2, 3))
            - (image_01[..., lower_start:, :] > 0.96).float().mean(dim=(1, 2, 3))
        ).clamp_min(0.0)
        batch_size = image.shape[0]
        totals["mae"] += mae.sum().item()
        totals["ssim"] += ssim.sum().item()
        totals["dark_field_dice"] += dark_dice.sum().item()
        totals["lower_saturation_excess"] += lower_sat.sum().item()
        count += batch_size
        for offset in range(batch_size):
            record = {
                "path": batch["path"][offset],
                "subject_id": batch["subject_id"][offset],
                "dicom_id": batch["dicom_id"][offset],
                "mae": float(mae[offset].item()),
                "ssim": float(ssim[offset].item()),
                "dark_field_dice": float(dark_dice[offset].item()),
                "lower_saturation_excess": float(lower_sat[offset].item()),
                "image": image_01[offset, 0].detach().cpu(),
                "reconstruction": reconstruction_01[offset, 0].detach().cpu(),
                "abs_error": (reconstruction_01[offset, 0] - image_01[offset, 0])
                .abs()
                .detach()
                .cpu(),
            }
            if len(random_records) < args.sample_count:
                random_records.append(record)
            worst_records.append(record)
            worst_records = sorted(
                worst_records,
                key=lambda item: item["mae"],
                reverse=True,
            )[: args.sample_count]

    summary = {key: value / max(count, 1) for key, value in totals.items()}
    summary["count"] = int(count)
    summary["passes_qc_gate"] = bool(
        summary["ssim"] >= 0.80
        and summary["mae"] <= 0.055
        and summary["dark_field_dice"] >= 0.90
        and summary["lower_saturation_excess"] <= 0.002
    )
    return summary, random_records, worst_records


def save_panel(records, output_path, title, sample_count):
    if not records:
        return
    selected = records[:sample_count]
    fig, axes = plt.subplots(3, len(selected), figsize=(4 * len(selected), 9))
    if len(selected) == 1:
        axes = axes.reshape(3, 1)
    for col, record in enumerate(selected):
        images = [
            (record["image"], f"Input\nSSIM={record['ssim']:.3f}"),
            (record["reconstruction"], "Reconstruction"),
            (record["abs_error"], f"Abs error\nMAE={record['mae']:.3f}"),
        ]
        for row, (image, label) in enumerate(images):
            axes[row, col].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, col].set_title(label)
            axes[row, col].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_checkpoint(path, epoch, score, model, ema, optimizer, scaler, config, summary):
    checkpoint = {
        "epoch": epoch,
        "score": score,
        "config": config,
        "summary": summary,
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema.ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
    }
    torch.save(checkpoint, path)


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
    train_dataset = CXRImageDataset(
        Path(args.manifest_dir) / "train.csv",
        args.image_size,
        verify_readable=not args.skip_image_verify,
        foreground_crop=args.foreground_crop,
        crop_threshold=args.crop_threshold,
        crop_min_content_fraction=args.crop_min_content_fraction,
        crop_margin_fraction=args.crop_margin_fraction,
    )
    val_dataset = CXRImageDataset(
        Path(args.manifest_dir) / "val.csv",
        args.image_size,
        verify_readable=not args.skip_image_verify,
        foreground_crop=args.foreground_crop,
        crop_threshold=args.crop_threshold,
        crop_min_content_fraction=args.crop_min_content_fraction,
        crop_margin_fraction=args.crop_margin_fraction,
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "use_amp": use_amp,
            "save_path": str(save_path),
            "output_dir": str(output_dir),
            "sample_dir": str(sample_dir),
        }
    )
    model = CXRLatentAutoencoder(
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
    ).to(device)
    print(f"Autoencoder parameters: {sum(p.numel() for p in model.parameters()):,}")
    ema = ExponentialMovingAverage(model, args.ema_decay)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    ssim_window = create_ssim_window(11, 1.5, device)

    best_score = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {}
        for batch in tqdm(train_loader, desc=f"AE epoch {epoch}/{args.epochs}"):
            batch["image"] = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, metrics = autoencoder_loss(model, batch, ssim_window, args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value.item())

        train_summary = {key: value / max(len(train_loader), 1) for key, value in running.items()}
        val_summary, val_records, worst_records = evaluate(ema.ema_model, val_loader, device, args)
        score = val_summary["mae"] + max(0.0, 0.80 - val_summary["ssim"])
        row = {"epoch": epoch, "train": train_summary, "val": val_summary, "score": score}
        history.append(row)
        (output_dir / "autoencoder_history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(row, indent=2))

        save_panel(
            val_records,
            sample_dir / f"epoch_{epoch:03d}_random_panel.png",
            f"Autoencoder epoch {epoch} random validation",
            args.sample_count,
        )
        save_panel(
            worst_records,
            sample_dir / f"epoch_{epoch:03d}_worst_panel.png",
            f"Autoencoder epoch {epoch} worst validation",
            args.sample_count,
        )

        if score < best_score:
            best_score = score
            save_checkpoint(
                save_path,
                epoch,
                score,
                model,
                ema,
                optimizer,
                scaler,
                config,
                val_summary,
            )
            val_summary_with_epoch = dict(val_summary)
            val_summary_with_epoch["best_epoch"] = epoch
            (output_dir / "autoencoder_eval_summary.json").write_text(
                json.dumps(val_summary_with_epoch, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Saved best autoencoder checkpoint to {save_path}")


if __name__ == "__main__":
    main()
