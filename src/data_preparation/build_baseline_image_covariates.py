#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, UnidentifiedImageError
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build first-pass baseline image covariates for the target-trial cohort"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resize", type=int, default=32)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--output-manifest", default=None)
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "study_name" not in config:
        raise ValueError(f"Invalid target-trial config: {config_path}")
    return config


def default_output_dir(config):
    final_root = Path(__file__).resolve().parents[2]
    return final_root / "results" / config["study_name"]


def infer_pairs_path(output_dir):
    candidates = sorted(output_dir.glob("target_trial_pairs*.csv"))
    candidates = [path for path in candidates if "screening" not in path.name]
    if not candidates:
        raise FileNotFoundError(
            f"Could not infer target-trial pairs CSV in {output_dir}. "
            "Pass --pairs-path explicitly."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous target-trial pair files in {output_dir}: {[str(path.name) for path in candidates]}"
        )
    return candidates[0]


def coerce_bool(series):
    if str(series.dtype) == "bool":
        return series.fillna(False)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    return series.astype("string").str.strip().str.lower().map(mapping).fillna(False)


def load_and_resize_image(path, resize):
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((resize, resize), resample=Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def gradient_energy(array):
    gy, gx = np.gradient(array)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


def center_mean(array):
    h, w = array.shape
    h0, h1 = int(h * 0.25), int(h * 0.75)
    w0, w1 = int(w * 0.25), int(w * 0.75)
    return float(array[h0:h1, w0:w1].mean())


def extract_scalar_features(array):
    values = array.ravel()
    return {
        "img_mean": float(values.mean()),
        "img_std": float(values.std()),
        "img_min": float(values.min()),
        "img_max": float(values.max()),
        "img_p01": float(np.quantile(values, 0.01)),
        "img_p05": float(np.quantile(values, 0.05)),
        "img_p50": float(np.quantile(values, 0.50)),
        "img_p95": float(np.quantile(values, 0.95)),
        "img_p99": float(np.quantile(values, 0.99)),
        "img_frac_below_005": float((values < 0.05).mean()),
        "img_frac_above_095": float((values > 0.95).mean()),
        "img_gradient_energy": gradient_energy(array),
        "img_center_mean": center_mean(array),
        "img_center_minus_global_mean": center_mean(array) - float(values.mean()),
    }


def build_feature_matrix(pairs, resize, progress_every):
    rows = []
    embeddings = []
    missing_rows = []
    total = len(pairs)

    for index, row in enumerate(pairs.itertuples(index=False), start=1):
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total):
            print(f"[image_covariates] processed {index}/{total} baseline images", flush=True)
        path = Path(str(row.path_0))
        if not bool(row.image_exists_0) or str(path) in {"", "<NA>", "nan"} or not path.exists():
            missing_rows.append(
                {
                    "subject_id": row.subject_id,
                    "hadm_id": row.hadm_id,
                    "study_id_0": row.study_id_0,
                    "cxr_0": row.cxr_0,
                    "path_0": str(path),
                    "missing_reason": "missing_path_or_file",
                }
            )
            continue

        try:
            image = load_and_resize_image(path, resize=resize)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            missing_rows.append(
                {
                    "subject_id": row.subject_id,
                    "hadm_id": row.hadm_id,
                    "study_id_0": row.study_id_0,
                    "cxr_0": row.cxr_0,
                    "path_0": str(path),
                    "missing_reason": f"image_decode_error:{type(exc).__name__}",
                }
            )
            continue
        scalar_features = extract_scalar_features(image)
        rows.append(
            {
                "subject_id": row.subject_id,
                "hadm_id": row.hadm_id,
                "study_id_0": row.study_id_0,
                "cxr_0": row.cxr_0,
                "path_0": str(path),
                **scalar_features,
            }
        )
        embeddings.append(image.reshape(-1))

    feature_table = pd.DataFrame(rows)
    if embeddings:
        embedding_matrix = np.stack(embeddings).astype(np.float32, copy=False)
    else:
        embedding_matrix = np.zeros((0, resize * resize), dtype=np.float32)
    missing_table = pd.DataFrame(missing_rows)
    return feature_table, embedding_matrix, missing_table


def add_pca_features(feature_table, embedding_matrix, requested_components):
    if feature_table.empty:
        return feature_table, {"n_components": 0, "explained_variance_ratio": []}

    max_components = min(
        int(requested_components),
        int(embedding_matrix.shape[0]),
        int(embedding_matrix.shape[1]),
    )
    if max_components <= 0:
        return feature_table, {"n_components": 0, "explained_variance_ratio": []}

    pca = PCA(n_components=max_components, svd_solver="randomized", random_state=42)
    transformed = pca.fit_transform(embedding_matrix)
    for index in range(max_components):
        feature_table[f"img_pca_{index + 1:02d}"] = transformed[:, index]

    metadata = {
        "n_components": max_components,
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }
    return feature_table, metadata


def build_baseline_image_covariates(pairs, resize, pca_components, progress_every):
    table = pairs.copy()
    table["image_exists_0"] = coerce_bool(table["image_exists_0"])
    feature_table, embedding_matrix, missing_table = build_feature_matrix(
        table,
        resize=resize,
        progress_every=progress_every,
    )
    feature_table, pca_metadata = add_pca_features(
        feature_table,
        embedding_matrix,
        requested_components=pca_components,
    )
    return feature_table, missing_table, pca_metadata


def summarize_outputs(pairs, features, missing, pca_metadata, resize):
    summary = {
        "input_rows": int(len(pairs)),
        "image_rows_with_features": int(len(features)),
        "image_rows_missing": int(len(missing)),
        "resize": int(resize),
        "scalar_feature_columns": [
            column
            for column in features.columns
            if column.startswith("img_") and not column.startswith("img_pca_")
        ],
        "pca": pca_metadata,
    }
    if not features.empty:
        summary["feature_means"] = {
            column: float(features[column].mean())
            for column in features.columns
            if column.startswith("img_") and not column.startswith("img_pca_")
        }
    return summary


def write_outputs(output_dir, features, missing, summary, output_manifest):
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "baseline_image_covariates.csv"
    missing_path = output_dir / "baseline_image_covariates_missing.csv"
    summary_path = output_dir / "baseline_image_covariates_summary.json"

    features.to_csv(features_path, index=False)
    missing.to_csv(missing_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Saved baseline image covariates to {features_path}")
    print(f"Saved missing-image audit to {missing_path}")
    print(f"Saved summary to {summary_path}")
    print(
        "[summary] "
        f"rows={summary['input_rows']} "
        f"features={summary['image_rows_with_features']} "
        f"missing={summary['image_rows_missing']} "
        f"pca_components={summary['pca']['n_components']}"
    )

    if output_manifest:
        manifest_path = Path(output_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "script": "build_baseline_image_covariates.py",
            "features_path": str(features_path),
            "missing_path": str(missing_path),
            "summary_path": str(summary_path),
            "rows": summary["input_rows"],
            "feature_rows": summary["image_rows_with_features"],
        }
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(config)
    pairs_path = Path(args.pairs_path) if args.pairs_path else infer_pairs_path(output_dir)
    pairs = pd.read_csv(
        pairs_path,
        usecols=["subject_id", "hadm_id", "study_id_0", "cxr_0", "path_0", "image_exists_0"],
        low_memory=False,
    )

    features, missing, pca_metadata = build_baseline_image_covariates(
        pairs,
        resize=args.resize,
        pca_components=args.pca_components,
        progress_every=args.progress_every,
    )
    summary = summarize_outputs(
        pairs,
        features=features,
        missing=missing,
        pca_metadata=pca_metadata,
        resize=args.resize,
    )
    write_outputs(output_dir, features, missing, summary, args.output_manifest)


if __name__ == "__main__":
    main()
