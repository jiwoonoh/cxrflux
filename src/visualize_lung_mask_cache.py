#!/usr/bin/env python3
"""Create a quick overlay panel for a cached lung-mask directory."""

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms

from dataset import ForegroundCropAndPad
from mask_utils import load_lung_mask


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize cached CXR lung masks.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--split-dir", default="")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--foreground-crop", action="store_true")
    parser.add_argument("--crop-threshold", type=int, default=10)
    parser.add_argument("--crop-min-content-fraction", type=float, default=0.02)
    parser.add_argument("--crop-margin-fraction", type=float, default=0.03)
    return parser.parse_args()


def read_frame(csv_path, split_dir):
    split_dir = Path(split_dir) if split_dir else None
    if split_dir and (split_dir / "test.csv").exists():
        frame = pd.read_csv(split_dir / "test.csv")
    else:
        frame = pd.read_csv(csv_path)
    if "cxr_0" not in frame.columns and "dicom_id" in frame.columns:
        frame = frame.rename(columns={"dicom_id": "cxr_0"})
    if "cxr_0" not in frame.columns:
        raise ValueError("Input table must contain cxr_0 or dicom_id")
    return frame.dropna(subset=["cxr_0"]).copy()


def build_image_transform(args):
    steps = []
    if args.foreground_crop:
        steps.append(
            ForegroundCropAndPad(
                threshold=args.crop_threshold,
                min_content_fraction=args.crop_min_content_fraction,
                margin_fraction=args.crop_margin_fraction,
            )
        )
    steps.append(transforms.Resize((args.image_size, args.image_size)))
    return transforms.Compose(steps)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    frame = read_frame(args.csv_path, args.split_dir)
    dicom_ids = list(dict.fromkeys(frame["cxr_0"].astype(str).tolist()))
    rng.shuffle(dicom_ids)
    dicom_ids = dicom_ids[: args.sample_count]
    image_transform = build_image_transform(args)

    fig, axes = plt.subplots(2, len(dicom_ids), figsize=(3.0 * len(dicom_ids), 6.0))
    if len(dicom_ids) == 1:
        axes = axes.reshape(2, 1)
    for col, dicom_id in enumerate(dicom_ids):
        image_path = Path(args.image_root) / f"{dicom_id}.jpg"
        with Image.open(image_path) as image:
            image = image_transform(image.convert("L"))
        mask = load_lung_mask(args.mask_root, dicom_id, image_size=args.image_size)[0].numpy()

        axes[0, col].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(dicom_id[:8])
        axes[0, col].axis("off")

        axes[1, col].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[1, col].imshow(mask, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.35)
        axes[1, col].axis("off")

    fig.tight_layout()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
