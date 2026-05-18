# dataset.py

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from mask_utils import load_lung_mask


DEFAULT_SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ForegroundCropAndPad:
    """Deterministically remove large black acquisition borders before resizing."""

    def __init__(
        self,
        threshold=10,
        min_content_fraction=0.02,
        margin_fraction=0.03,
        fill=0,
        max_scan_size=512,
    ):
        self.threshold = int(threshold)
        self.min_content_fraction = float(min_content_fraction)
        self.margin_fraction = float(margin_fraction)
        self.fill = int(fill)
        self.max_scan_size = int(max_scan_size)

    def __call__(self, image):
        width, height = image.size
        scan_width = min(width, self.max_scan_size)
        scan_height = max(1, round(height * scan_width / max(width, 1)))
        grayscale = image.convert("L").resize((scan_width, scan_height))
        pixels = grayscale.load()

        row_counts = [0] * scan_height
        col_counts = [0] * scan_width
        for y in range(scan_height):
            row_count = 0
            for x in range(scan_width):
                if pixels[x, y] > self.threshold:
                    row_count += 1
                    col_counts[x] += 1
            row_counts[y] = row_count

        min_row_count = max(1, int(round(scan_width * self.min_content_fraction)))
        min_col_count = max(1, int(round(scan_height * self.min_content_fraction)))
        rows = [index for index, count in enumerate(row_counts) if count >= min_row_count]
        cols = [index for index, count in enumerate(col_counts) if count >= min_col_count]
        if not rows or not cols:
            return image

        scale_x = width / scan_width
        scale_y = height / scan_height
        left = int(min(cols) * scale_x)
        right = int((max(cols) + 1) * scale_x)
        top = int(min(rows) * scale_y)
        bottom = int((max(rows) + 1) * scale_y)

        margin = int(round(max(right - left, bottom - top) * self.margin_fraction))
        left = max(0, left - margin)
        right = min(width, right + margin)
        top = max(0, top - margin)
        bottom = min(height, bottom + margin)

        cropped = image.crop((left, top, right, bottom))
        crop_width, crop_height = cropped.size
        side = max(crop_width, crop_height)
        canvas = Image.new(image.mode, (side, side), color=self.fill)
        paste_left = (side - crop_width) // 2
        paste_top = (side - crop_height) // 2
        canvas.paste(cropped, (paste_left, paste_top))
        return canvas


class CXRPairDataset(Dataset):
    def __init__(
        self,
        pairs_df,
        image_root,
        transform=None,
        weight_column=None,
        propensity_column=None,
        treatment_column="treated",
        lung_mask_root=None,
        extra_numeric_columns=None,
    ):
        self.pairs = pairs_df.reset_index(drop=True).copy()
        self.image_root = image_root
        self.transform = transform
        self.weight_column = weight_column
        self.propensity_column = propensity_column
        self.treatment_column = treatment_column or "treated"
        self.lung_mask_root = lung_mask_root
        self.extra_numeric_columns = list(extra_numeric_columns or [])

    def _resolve_image_path(self, value):
        path = Path(str(value))
        if path.is_absolute() and path.exists():
            return path
        candidates = []
        if not path.is_absolute():
            candidates.append(path)
        if self.image_root:
            root = Path(self.image_root)
            candidates.append(root / path)
            candidates.append(root / path.name)
        else:
            root = None
        if path.is_absolute() and root is not None:
            candidates.append(root / path.name)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if root is not None:
            return root / path.name
        return path

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]

        path_0 = self._resolve_image_path(row["path_0"])
        path_1 = self._resolve_image_path(row["path_1"])

        try:
            img_0 = Image.open(path_0).convert("L")
            img_1 = Image.open(path_1).convert("L")
            img_0.load()
            img_1.load()
        except (OSError, IOError) as exc:
            raise RuntimeError(
                f"Failed to load image pair {row['cxr_0']} / {row['cxr_1']}"
            ) from exc

        if self.transform:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)

        if self.treatment_column not in row.index:
            pair_id = row["pair_id"] if "pair_id" in row.index else idx
            raise KeyError(
                f"Missing treatment column '{self.treatment_column}' in pair {pair_id}"
            )
        treatment_value = pd.to_numeric(row[self.treatment_column], errors="coerce")
        if pd.isna(treatment_value):
            pair_id = row["pair_id"] if "pair_id" in row.index else idx
            raise ValueError(
                f"Invalid treatment value for pair {pair_id} in column "
                f"'{self.treatment_column}'"
            )

        sample = {
            "x_0": img_0,
            "y": img_1,
            "delta": torch.tensor([row["delta_normalized"]], dtype=torch.float32),
            "a": torch.tensor([float(treatment_value)], dtype=torch.float32),
            "subject_id": row["subject_id"],
            "cxr_0": row["cxr_0"],
            "cxr_1": row["cxr_1"],
            "hours_diff": torch.tensor([row["hours_diff"]], dtype=torch.float32),
        }
        if self.weight_column:
            if self.weight_column not in row.index:
                raise KeyError(f"Missing weight column '{self.weight_column}' in dataset row")
            weight_value = pd.to_numeric(row[self.weight_column], errors="coerce")
            if pd.isna(weight_value):
                pair_id = row["pair_id"] if "pair_id" in row.index else idx
                raise ValueError(
                    f"Invalid weight value for pair {pair_id} in column '{self.weight_column}'"
                )
            sample["sample_weight"] = torch.tensor([float(weight_value)], dtype=torch.float32)
        if self.propensity_column:
            if self.propensity_column not in row.index:
                raise KeyError(
                    f"Missing propensity column '{self.propensity_column}' in dataset row"
                )
            propensity_value = pd.to_numeric(row[self.propensity_column], errors="coerce")
            if pd.isna(propensity_value):
                pair_id = row["pair_id"] if "pair_id" in row.index else idx
                raise ValueError(
                    f"Invalid propensity value for pair {pair_id} in column "
                    f"'{self.propensity_column}'"
                )
            sample["propensity"] = torch.tensor(
                [float(propensity_value)],
                dtype=torch.float32,
            )
        if self.lung_mask_root:
            lung_mask = load_lung_mask(self.lung_mask_root, row["cxr_0"])
            if tuple(lung_mask.shape[-2:]) != tuple(img_0.shape[-2:]):
                lung_mask = F.interpolate(
                    lung_mask.unsqueeze(0),
                    size=img_0.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            sample["lung_mask"] = lung_mask.float().clamp(0.0, 1.0)
        for column in self.extra_numeric_columns:
            if column not in row.index:
                raise KeyError(f"Missing extra numeric column '{column}' in dataset row")
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.isna(value):
                value = 0.0
            sample[column] = torch.tensor([float(value)], dtype=torch.float32)
        return sample


def build_transform(
    image_size,
    foreground_crop=False,
    crop_threshold=10,
    crop_min_content_fraction=0.02,
    crop_margin_fraction=0.03,
):
    transform_steps = []
    if foreground_crop:
        transform_steps.append(
            ForegroundCropAndPad(
                threshold=crop_threshold,
                min_content_fraction=crop_min_content_fraction,
                margin_fraction=crop_margin_fraction,
            )
        )
    transform_steps.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return transforms.Compose(transform_steps)


def _coerce_treated(series):
    lowered = series.astype(str).str.lower()
    mapped = lowered.map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})

    if mapped.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        mapped = mapped.fillna(numeric)

    return mapped.astype(float)


@lru_cache(maxsize=None)
def _is_readable_image(path):
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, IOError):
        return False


def _standardize_pairs_dataframe(pairs, image_root, verify_readable=True):
    pairs = pairs.copy()

    rename_map = {}
    if "cxr_0" not in pairs.columns and "dicom_id" in pairs.columns:
        rename_map["dicom_id"] = "cxr_0"
    if "cxr_1" not in pairs.columns and "next_dicom_id" in pairs.columns:
        rename_map["next_dicom_id"] = "cxr_1"
    if "hours_diff" not in pairs.columns and "time_gap_hrs" in pairs.columns:
        rename_map["time_gap_hrs"] = "hours_diff"
    if "treated" not in pairs.columns and "is_treated" in pairs.columns:
        rename_map["is_treated"] = "treated"

    if rename_map:
        pairs = pairs.rename(columns=rename_map)

    required_columns = ["subject_id", "cxr_0", "cxr_1", "hours_diff", "treated"]
    missing_columns = [column for column in required_columns if column not in pairs.columns]
    if missing_columns:
        raise ValueError(
            f"Pairs CSV is missing required columns: {', '.join(missing_columns)}"
        )

    pairs["subject_id"] = pairs["subject_id"].astype(str)
    pairs["cxr_0"] = pairs["cxr_0"].astype(str)
    pairs["cxr_1"] = pairs["cxr_1"].astype(str)
    pairs["hours_diff"] = pd.to_numeric(pairs["hours_diff"], errors="coerce")
    pairs["treated"] = _coerce_treated(pairs["treated"])

    pairs = pairs.dropna(subset=["subject_id", "cxr_0", "cxr_1", "hours_diff", "treated"])
    pairs["path_0"] = pairs["cxr_0"].map(
        lambda dicom_id: str(Path(image_root) / f"{dicom_id}.jpg")
    )
    pairs["path_1"] = pairs["cxr_1"].map(
        lambda dicom_id: str(Path(image_root) / f"{dicom_id}.jpg")
    )
    pairs["delta_normalized"] = pairs["hours_diff"] / 48.0

    pairs = pairs[(pairs["delta_normalized"] >= 0.0) & (pairs["delta_normalized"] <= 1.0)]

    exists_mask = pairs["path_0"].map(os.path.exists) & pairs["path_1"].map(os.path.exists)
    dropped_missing_pairs = int((~exists_mask).sum())
    if dropped_missing_pairs:
        print(f"[data] Dropped {dropped_missing_pairs} pairs with missing images before splitting")

    pairs = pairs.loc[exists_mask].reset_index(drop=True)

    if verify_readable:
        unique_paths = pd.unique(pd.concat([pairs["path_0"], pairs["path_1"]], ignore_index=True))
        readable_lookup = {path: _is_readable_image(path) for path in unique_paths}
        readable_mask = pairs["path_0"].map(readable_lookup) & pairs["path_1"].map(readable_lookup)
        dropped_corrupt_pairs = int((~readable_mask).sum())
        if dropped_corrupt_pairs:
            corrupted_files = sum(1 for is_readable in readable_lookup.values() if not is_readable)
            print(
                f"[data] Dropped {dropped_corrupt_pairs} pairs with unreadable images "
                f"({corrupted_files} corrupted files) before splitting"
            )
        pairs = pairs.loc[readable_mask].reset_index(drop=True)
    else:
        print("[data] Skipping full image readability verification")

    return pairs.reset_index(drop=True)


def _split_pairs_by_subject(pairs, split_fractions, seed):
    train_fraction, val_fraction, test_fraction = split_fractions
    total_fraction = train_fraction + val_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    subject_ids = (
        pairs["subject_id"].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    )
    subject_count = len(subject_ids)
    if subject_count < 3:
        raise ValueError("Need at least 3 subjects to create train/val/test splits")

    train_end = int(subject_count * train_fraction)
    val_end = train_end + int(subject_count * val_fraction)

    train_end = max(1, train_end)
    val_end = max(train_end + 1, val_end)
    val_end = min(val_end, subject_count - 1)

    train_subjects = set(subject_ids[:train_end])
    val_subjects = set(subject_ids[train_end:val_end])
    test_subjects = set(subject_ids[val_end:])

    if not val_subjects or not test_subjects:
        raise ValueError("Subject split produced an empty validation or test set")

    split_frames = {
        "train": pairs[pairs["subject_id"].isin(train_subjects)].reset_index(drop=True),
        "val": pairs[pairs["subject_id"].isin(val_subjects)].reset_index(drop=True),
        "test": pairs[pairs["subject_id"].isin(test_subjects)].reset_index(drop=True),
    }
    return split_frames


def _save_split_frames(split_frames, split_dir):
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, frame in split_frames.items():
        frame.to_csv(split_dir / f"{split_name}.csv", index=False)


def _load_saved_split_frames(split_dir, image_root, verify_readable=True):
    split_dir = Path(split_dir)
    split_paths = {name: split_dir / f"{name}.csv" for name in ("train", "val", "test")}
    if not all(path.exists() for path in split_paths.values()):
        return None

    split_frames = {
        split_name: _standardize_pairs_dataframe(
            pd.read_csv(split_path),
            image_root,
            verify_readable=verify_readable,
        )
        for split_name, split_path in split_paths.items()
    }
    print(f"[data] Loaded saved subject-level splits from {split_dir}")
    return split_frames


def prepare_pair_splits(
    csv_path,
    image_root,
    split_dir=None,
    seed=42,
    split_fractions=DEFAULT_SPLIT_FRACTIONS,
    verify_readable=True,
):
    if split_dir:
        split_frames = _load_saved_split_frames(
            split_dir,
            image_root,
            verify_readable=verify_readable,
        )
        if split_frames is not None:
            _print_split_stats(split_frames)
            return split_frames

    pairs = _standardize_pairs_dataframe(
        pd.read_csv(csv_path),
        image_root,
        verify_readable=verify_readable,
    )
    split_frames = _split_pairs_by_subject(pairs, split_fractions, seed)

    if split_dir:
        _save_split_frames(split_frames, split_dir)
        print(f"[data] Saved subject-level splits to {split_dir}")

    _print_split_stats(split_frames)
    return split_frames


def _print_split_stats(split_frames):
    for split_name, frame in split_frames.items():
        pair_count = len(frame)
        subject_count = frame["subject_id"].nunique()
        print(f"[{split_name}] Loaded {pair_count} pairs across {subject_count} subjects")


def get_dataloaders(
    csv_path,
    image_root,
    batch_size=16,
    image_size=128,
    num_workers=4,
    seed=42,
    split_dir=None,
    split_fractions=DEFAULT_SPLIT_FRACTIONS,
    weight_column=None,
    verify_readable=True,
    foreground_crop=False,
    crop_threshold=10,
    crop_min_content_fraction=0.02,
    crop_margin_fraction=0.03,
):
    split_frames = prepare_pair_splits(
        csv_path=csv_path,
        image_root=image_root,
        split_dir=split_dir,
        seed=seed,
        split_fractions=split_fractions,
        verify_readable=verify_readable,
    )

    transform = build_transform(
        image_size,
        foreground_crop=foreground_crop,
        crop_threshold=crop_threshold,
        crop_min_content_fraction=crop_min_content_fraction,
        crop_margin_fraction=crop_margin_fraction,
    )
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    datasets = {
        split_name: CXRPairDataset(frame, image_root, transform, weight_column=weight_column)
        for split_name, frame in split_frames.items()
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
    }

    return loaders, split_frames
