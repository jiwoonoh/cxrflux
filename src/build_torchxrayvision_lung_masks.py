#!/usr/bin/env python3
"""Build a cached CXR lung-mask directory using TorchXRayVision ChestX-Det.

The cache is keyed by DICOM id and stores one soft [1, image_size, image_size]
lung mask per image as a .pt tensor. The same deterministic foreground crop
used by the prediction model can be applied here, which keeps masks aligned
with generated images during treatment-effect localization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dataset import ForegroundCropAndPad


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="Cache TorchXRayVision lung segmentations.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-dir", default="")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--segmenter-size",
        type=int,
        default=512,
        help="Image size sent to the ChestX-Det segmenter before masks are resized to --image-size.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--foreground-crop", action="store_true")
    parser.add_argument("--crop-threshold", type=int, default=10)
    parser.add_argument("--crop-min-content-fraction", type=float, default=0.02)
    parser.add_argument("--crop-margin-fraction", type=float, default=0.03)
    parser.add_argument("--include-followup", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def standardize_id_columns(frame):
    rename_map = {}
    if "cxr_0" not in frame.columns and "dicom_id" in frame.columns:
        rename_map["dicom_id"] = "cxr_0"
    if "cxr_1" not in frame.columns and "next_dicom_id" in frame.columns:
        rename_map["next_dicom_id"] = "cxr_1"
    if rename_map:
        frame = frame.rename(columns=rename_map)
    if "cxr_0" not in frame.columns:
        raise ValueError("Input table must contain cxr_0 or dicom_id")
    return frame


def load_pair_frame(csv_path, split_dir):
    split_dir = Path(split_dir) if split_dir else None
    if split_dir and all((split_dir / f"{name}.csv").exists() for name in ("train", "val", "test")):
        frames = [pd.read_csv(split_dir / f"{name}.csv") for name in ("train", "val", "test")]
        return pd.concat(frames, ignore_index=True)
    return pd.read_csv(csv_path)


def collect_image_rows(args):
    frame = standardize_id_columns(load_pair_frame(args.csv_path, args.split_dir))
    ids = frame["cxr_0"].dropna().astype(str).tolist()
    if args.include_followup and "cxr_1" in frame.columns:
        ids.extend(frame["cxr_1"].dropna().astype(str).tolist())
    rows = []
    seen = set()
    for dicom_id in ids:
        if dicom_id in seen:
            continue
        seen.add(dicom_id)
        path = Path(args.image_root) / f"{dicom_id}.jpg"
        if path.exists():
            rows.append({"dicom_id": dicom_id, "path": str(path)})
    if args.max_images is not None:
        rows = rows[: args.max_images]
    return rows


class CXRImageDataset(Dataset):
    def __init__(
        self,
        rows,
        segmenter_size,
        foreground_crop,
        crop_threshold,
        crop_min_content_fraction,
        crop_margin_fraction,
    ):
        self.rows = rows
        self.segmenter_size = segmenter_size
        crop_steps = []
        if foreground_crop:
            crop_steps.append(
                ForegroundCropAndPad(
                    threshold=crop_threshold,
                    min_content_fraction=crop_min_content_fraction,
                    margin_fraction=crop_margin_fraction,
                )
            )
        crop_steps.append(transforms.Resize((segmenter_size, segmenter_size)))
        self.display_transform = transforms.Compose(crop_steps)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            image = self.display_transform(image.convert("L"))
        array = np.asarray(image, dtype=np.float32)
        return {
            "dicom_id": row["dicom_id"],
            "image_255": torch.from_numpy(array).unsqueeze(0),
        }


def find_lung_indices(seg_model):
    targets = getattr(seg_model, "targets", None)
    if not targets:
        return [4, 5]
    normalized = [str(target).lower().replace("_", " ") for target in targets]
    indices = [
        index
        for index, target in enumerate(normalized)
        if target in {"left lung", "right lung"}
    ]
    if len(indices) != 2:
        raise ValueError(f"Could not find left/right lung classes in TorchXRayVision targets: {targets}")
    return indices


def prepare_txv_input(image_255, xrv):
    images = []
    for image in image_255:
        array = image.squeeze(0).numpy()
        normalized = xrv.datasets.normalize(array, 255)
        images.append(torch.from_numpy(normalized).float().unsqueeze(0))
    return torch.stack(images, dim=0)


def main():
    args = parse_args()
    try:
        import torchxrayvision as xrv
    except ImportError as exc:
        raise SystemExit(
            "TorchXRayVision is not installed. Install it in the PyTorch environment "
            "before running this cache builder, for example: python -m pip install torchxrayvision"
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_image_rows(args)
    pending = [
        row
        for row in rows
        if args.overwrite or not (output_dir / f"{row['dicom_id']}.pt").exists()
    ]
    if not pending:
        print(json.dumps({"output_dir": str(output_dir), "created": 0, "skipped_existing": len(rows)}, indent=2))
        return

    dataset = CXRImageDataset(
        pending,
        segmenter_size=args.segmenter_size,
        foreground_crop=args.foreground_crop,
        crop_threshold=args.crop_threshold,
        crop_min_content_fraction=args.crop_min_content_fraction,
        crop_margin_fraction=args.crop_margin_fraction,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    lung_indices = find_lung_indices(seg_model)

    created = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="building lung-mask cache"):
            txv_input = prepare_txv_input(batch["image_255"], xrv).to(device)
            output = seg_model(txv_input)
            if isinstance(output, dict):
                output = output.get("out", next(value for value in output.values() if torch.is_tensor(value)))
            output = output.float()
            if float(output.min()) < 0.0 or float(output.max()) > 1.0:
                output = torch.sigmoid(output)
            lungs = output[:, lung_indices].amax(dim=1, keepdim=True).clamp(0.0, 1.0)
            if tuple(lungs.shape[-2:]) != (args.image_size, args.image_size):
                lungs = F.interpolate(
                    lungs,
                    size=(args.image_size, args.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            lungs = lungs.cpu().float().clamp(0.0, 1.0)
            for dicom_id, mask in zip(batch["dicom_id"], lungs):
                torch.save(mask, output_dir / f"{dicom_id}.pt")
                created += 1

    summary = {
        "output_dir": str(output_dir),
        "created": created,
        "skipped_existing": len(rows) - len(pending),
        "image_count_requested": len(rows),
        "image_size": args.image_size,
        "segmenter_size": args.segmenter_size,
        "foreground_crop": args.foreground_crop,
        "torchxrayvision_model": "baseline_models.chestx_det.PSPNet",
        "lung_indices": lung_indices,
    }
    (output_dir / "mask_cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
