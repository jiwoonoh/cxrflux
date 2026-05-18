#!/usr/bin/env python3

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml


FINAL_ROOT = Path(__file__).resolve().parents[2]
ROOT_COVARIATES_PATH = Path(__file__).resolve().with_name("build_causal_covariates.py")


def load_root_covariates_module():
    spec = importlib.util.spec_from_file_location(
        "root_build_causal_covariates",
        ROOT_COVARIATES_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load root covariate builder from {ROOT_COVARIATES_PATH}")
    spec.loader.exec_module(module)
    return module


root_cov = load_root_covariates_module()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build target-trial EHR covariates by reusing the validated HF covariate logic"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--patients-path",
        default="physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz",
    )
    parser.add_argument(
        "--admissions-path",
        default="physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz",
    )
    parser.add_argument(
        "--services-path",
        default="physionet.org/files/mimiciv/3.1/hosp/services.csv.gz",
    )
    parser.add_argument(
        "--icustays-path",
        default="physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz",
    )
    parser.add_argument(
        "--diagnoses-path",
        default="physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz",
    )
    parser.add_argument(
        "--diagnosis-titles-path",
        default="physionet.org/files/mimiciv/3.1/hosp/d_icd_diagnoses.csv.gz",
    )
    parser.add_argument(
        "--admin-events-path",
        default=(
            "data/cxr_pairs_frontal_iv_furosemide_heart_failure_admin_events.csv"
        ),
    )
    parser.add_argument(
        "--labitems-path",
        default="physionet.org/files/mimiciv/3.1/hosp/d_labitems.csv.gz",
    )
    parser.add_argument(
        "--labevents-path",
        default="physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz",
    )
    parser.add_argument(
        "--omr-path",
        default="physionet.org/files/mimiciv/3.1/hosp/omr.csv.gz",
    )
    parser.add_argument("--prior-furo-lookback-hours", type=float, default=72.0)
    parser.add_argument("--lab-lookback-hours", type=float, default=24.0)
    parser.add_argument("--omr-lookback-days", type=float, default=365.0)
    parser.add_argument("--chunksize", type=int, default=250000)
    parser.add_argument("--output-manifest", default=None)
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "study_name" not in config:
        raise ValueError(f"Invalid target-trial config: {config_path}")
    return config


def default_output_dir(config):
    return FINAL_ROOT / "results" / config["study_name"]


def infer_pairs_path(output_dir):
    candidates = sorted(output_dir.glob("target_trial_pairs*.csv"))
    candidates = [path for path in candidates if "screening" not in path.name]
    if not candidates:
        raise FileNotFoundError(
            f"Could not infer target-trial pairs CSV in {output_dir}. Pass --pairs-path explicitly."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous target-trial pair files in {output_dir}: {[str(path.name) for path in candidates]}"
        )
    return candidates[0]


def build_pair_id(frame):
    return (
        frame["subject_id"].astype("string")
        + "_"
        + frame["hadm_id"].astype("string")
        + "_"
        + frame["study_id_0"].astype("string")
    )


def load_target_trial_pairs(path):
    pairs = pd.read_csv(path, low_memory=False).copy()
    required = {"subject_id", "hadm_id", "study_id_0", "t0", "treated"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Missing required target-trial columns: {', '.join(sorted(missing))}")

    pairs["subject_id"] = root_cov.normalize_id_series(pairs["subject_id"])
    pairs["hadm_id"] = root_cov.normalize_id_series(pairs["hadm_id"])
    pairs["study_id_0"] = root_cov.normalize_id_series(pairs["study_id_0"])
    pairs["t0"] = pd.to_datetime(pairs["t0"], errors="coerce")
    pairs["treated"] = pairs["treated"].fillna(False).astype(bool)
    pairs = pairs.dropna(subset=["subject_id", "hadm_id", "study_id_0", "t0"]).copy()
    pairs["pair_id"] = build_pair_id(pairs)

    if pairs["pair_id"].duplicated().any():
        duplicates = int(pairs["pair_id"].duplicated().sum())
        raise ValueError(f"Target-trial pairs are not unique by pair_id; duplicates={duplicates}")

    return pairs


def build_covariates(pairs, args):
    demographic_features = root_cov.build_demographic_features(
        pairs,
        args.patients_path,
        args.admissions_path,
    )
    service_features = root_cov.build_service_features(pairs, args.services_path)
    icu_features = root_cov.build_icu_features(pairs, args.icustays_path)
    diagnosis_features = root_cov.build_diagnosis_features(
        pairs,
        args.diagnoses_path,
        args.diagnosis_titles_path,
        args.chunksize,
    )
    prior_furo_features = root_cov.build_prior_furo_features(
        pairs,
        args.admin_events_path,
        args.prior_furo_lookback_hours,
    )

    itemid_to_lab, used_itemids = root_cov.discover_lab_itemids(args.labitems_path)
    labevents = root_cov.load_target_labevents(
        args.labevents_path,
        pairs,
        itemid_to_lab,
        args.lab_lookback_hours,
        args.chunksize,
    )
    lab_features = root_cov.build_lab_features(pairs, labevents, args.lab_lookback_hours)
    omr_features = root_cov.build_omr_features(
        pairs,
        args.omr_path,
        args.omr_lookback_days,
        args.chunksize,
    )

    covariates = pairs[["pair_id"]].copy()
    for frame in [
        demographic_features,
        service_features,
        icu_features,
        diagnosis_features,
        prior_furo_features,
        lab_features,
        omr_features,
    ]:
        covariates = covariates.merge(frame, on="pair_id", how="left")

    causal_ready = pairs.merge(covariates, on="pair_id", how="left")
    summary = root_cov.build_summary(causal_ready, covariates, used_itemids)
    return covariates, causal_ready, summary


def write_outputs(output_dir, covariates, causal_ready, summary, output_manifest):
    output_dir.mkdir(parents=True, exist_ok=True)
    covariates_path = output_dir / "target_trial_causal_covariates.csv"
    causal_ready_path = output_dir / "target_trial_causal_ready.csv"
    summary_path = output_dir / "target_trial_causal_covariates_summary.json"

    covariates.to_csv(covariates_path, index=False)
    causal_ready.to_csv(causal_ready_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Saved target-trial covariates to {covariates_path}")
    print(f"Saved target-trial causal-ready table to {causal_ready_path}")
    print(f"Saved summary to {summary_path}")
    print(
        "[summary] "
        f"pairs={summary['n_pairs']} "
        f"treated={summary['treated_pairs']} "
        f"controls={summary['control_pairs']}"
    )

    if output_manifest:
        manifest_path = Path(output_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "script": "build_target_trial_covariates.py",
            "covariates_path": str(covariates_path),
            "causal_ready_path": str(causal_ready_path),
            "summary_path": str(summary_path),
            "counts": {
                "pairs": summary["n_pairs"],
                "treated": summary["treated_pairs"],
                "controls": summary["control_pairs"],
            },
        }
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(config)
    pairs_path = Path(args.pairs_path) if args.pairs_path else infer_pairs_path(output_dir)

    pairs = load_target_trial_pairs(pairs_path)
    covariates, causal_ready, summary = build_covariates(pairs, args)
    write_outputs(output_dir, covariates, causal_ready, summary, args.output_manifest)


if __name__ == "__main__":
    main()
