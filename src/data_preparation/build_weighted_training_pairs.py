#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach balancing weights to target-trial image pairs and write trainable split CSVs."
    )
    parser.add_argument("--pairs-path", required=True)
    parser.add_argument("--scores-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--source-split-dir", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--weight-column", default="balancing_weight")
    parser.add_argument("--observed-outcome-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_source_splits(source_split_dir):
    if source_split_dir is None:
        return None

    source_split_dir = Path(source_split_dir)
    split_frames = []
    for split_name in ("train", "val", "test"):
        split_path = source_split_dir / f"{split_name}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing source split file: {split_path}")
        split_frame = pd.read_csv(split_path, usecols=["pair_id"], low_memory=False)
        split_frame["pair_id"] = split_frame["pair_id"].astype(str)
        split_frame["split"] = split_name
        split_frames.append(split_frame)

    splits = pd.concat(split_frames, ignore_index=True)
    if splits["pair_id"].duplicated().any():
        duplicated = splits.loc[splits["pair_id"].duplicated(), "pair_id"].head(5).tolist()
        raise ValueError(f"Source splits contain duplicate pair_id values: {duplicated}")
    return splits


def assign_subject_splits(frame, seed):
    subject_frame = (
        frame.assign(
            subject_id=frame["subject_id"].astype(str),
            treated_numeric=pd.to_numeric(frame["treated"], errors="coerce").fillna(0),
        )
        .groupby("subject_id", as_index=False)["treated_numeric"]
        .max()
    )
    if len(subject_frame) < 3:
        raise ValueError("Need at least 3 subjects for train/val/test splits")

    def split_subjects(subject_ids):
        subject_ids = list(subject_ids)
        shuffled = pd.Series(subject_ids).sample(frac=1.0, random_state=seed).tolist()
        if len(shuffled) < 3:
            return {"train": shuffled, "val": [], "test": []}
        train_end = max(1, int(len(shuffled) * 0.8))
        val_end = max(train_end + 1, train_end + int(len(shuffled) * 0.1))
        val_end = min(val_end, len(shuffled) - 1)
        return {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

    treated_splits = split_subjects(subject_frame.loc[subject_frame["treated_numeric"] > 0, "subject_id"])
    control_splits = split_subjects(subject_frame.loc[subject_frame["treated_numeric"] <= 0, "subject_id"])

    split_lookup = {}
    for split_name in ("train", "val", "test"):
        for subject_id in treated_splits[split_name] + control_splits[split_name]:
            split_lookup[subject_id] = split_name

    if "val" not in split_lookup.values() or "test" not in split_lookup.values():
        raise ValueError("Subject split produced an empty validation or test set")
    return frame["subject_id"].astype(str).map(split_lookup)


def summarize_weight(frame, weight_column):
    weights = pd.to_numeric(frame[weight_column], errors="coerce")
    return {
        "mean": float(weights.mean()),
        "p50": float(weights.quantile(0.50)),
        "p95": float(weights.quantile(0.95)),
        "p99": float(weights.quantile(0.99)),
        "max": float(weights.max()),
        "positive_count": int((weights > 0).sum()),
        "near_zero_count": int((weights <= 1e-8).sum()),
    }


def main():
    args = parse_args()

    pairs = pd.read_csv(args.pairs_path, low_memory=False)
    scores = pd.read_csv(args.scores_path, low_memory=False)

    if "pair_id" not in pairs.columns and {"subject_id", "hadm_id", "study_id_0"}.issubset(pairs.columns):
        pairs["pair_id"] = (
            pairs["subject_id"].astype(str)
            + "_"
            + pairs["hadm_id"].astype(str)
            + "_"
            + pairs["study_id_0"].astype(str)
        )

    required_pair_columns = {"pair_id", "subject_id", "cxr_0", "cxr_1", "hours_diff", "treated"}
    missing_pair_columns = required_pair_columns - set(pairs.columns)
    if missing_pair_columns:
        raise ValueError(f"Pairs file missing columns: {', '.join(sorted(missing_pair_columns))}")

    required_score_columns = {"pair_id", args.weight_column}
    missing_score_columns = required_score_columns - set(scores.columns)
    if missing_score_columns:
        raise ValueError(f"Scores file missing columns: {', '.join(sorted(missing_score_columns))}")

    pairs = pairs.copy()
    scores = scores.copy()
    pairs["pair_id"] = pairs["pair_id"].astype(str)
    scores["pair_id"] = scores["pair_id"].astype(str)

    if scores["pair_id"].duplicated().any():
        duplicated = scores.loc[scores["pair_id"].duplicated(), "pair_id"].head(5).tolist()
        raise ValueError(f"Scores file contains duplicate pair_id values: {duplicated}")

    if args.observed_outcome_only:
        if "follow_up_image_observed" in pairs.columns:
            observed = pairs["follow_up_image_observed"].astype(str).str.lower().isin(["true", "1"])
            pairs = pairs.loc[observed].copy()
        pairs = pairs.dropna(subset=["cxr_1", "hours_diff"]).copy()

    merged = pairs.merge(
        scores[["pair_id", args.weight_column]],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    missing_weights = int(merged[args.weight_column].isna().sum())
    if missing_weights:
        raise ValueError(f"Missing {args.weight_column} for {missing_weights} pair rows")

    merged[args.weight_column] = pd.to_numeric(merged[args.weight_column], errors="coerce")
    if merged[args.weight_column].isna().any():
        raise ValueError(f"Non-numeric values found in {args.weight_column}")

    source_splits = load_source_splits(args.source_split_dir)
    if source_splits is not None:
        merged = merged.merge(source_splits, on="pair_id", how="left", validate="one_to_one")
        missing_split = int(merged["split"].isna().sum())
        if missing_split:
            raise ValueError(f"Missing source split assignment for {missing_split} rows")
    else:
        merged["split"] = assign_subject_splits(merged, args.seed)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    split_dir = Path(args.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {}
    for split_name in ("train", "val", "test"):
        split_frame = merged.loc[merged["split"] == split_name].copy()
        split_counts[split_name] = {
            "pairs": int(len(split_frame)),
            "subjects": int(split_frame["subject_id"].astype(str).nunique()),
            "treated": int(pd.to_numeric(split_frame["treated"], errors="coerce").fillna(0).sum()),
            "controls": int((pd.to_numeric(split_frame["treated"], errors="coerce").fillna(0) == 0).sum()),
            "weight": summarize_weight(split_frame, args.weight_column) if len(split_frame) else {},
        }
        split_frame.to_csv(split_dir / f"{split_name}.csv", index=False)

    summary = {
        "pairs_path": str(Path(args.pairs_path)),
        "scores_path": str(Path(args.scores_path)),
        "output_csv": str(output_csv),
        "split_dir": str(split_dir),
        "source_split_dir": args.source_split_dir,
        "weight_column": args.weight_column,
        "observed_outcome_only": args.observed_outcome_only,
        "n_pairs": int(len(merged)),
        "n_subjects": int(merged["subject_id"].astype(str).nunique()),
        "treated_pairs": int(pd.to_numeric(merged["treated"], errors="coerce").fillna(0).sum()),
        "control_pairs": int((pd.to_numeric(merged["treated"], errors="coerce").fillna(0) == 0).sum()),
        "weight": summarize_weight(merged, args.weight_column),
        "split_counts": split_counts,
    }

    summary_path = Path(args.summary_path) if args.summary_path else output_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved weighted training CSV to {output_csv}")
    print(f"Saved split CSVs to {split_dir}")
    print(f"Saved summary to {summary_path}")
    print(
        "[summary] "
        f"pairs={summary['n_pairs']} treated={summary['treated_pairs']} "
        f"controls={summary['control_pairs']} train={split_counts['train']['pairs']} "
        f"val={split_counts['val']['pairs']} test={split_counts['test']['pairs']}"
    )


if __name__ == "__main__":
    main()
