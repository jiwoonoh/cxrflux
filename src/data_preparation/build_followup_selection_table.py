#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build follow-up observation/selection table for the target-trial cohort"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs-path", default=None)
    parser.add_argument("--output-dir", default=None)
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
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map(mapping).fillna(False)


def derive_observation_status(frame):
    status = pd.Series("no_follow_up_study_in_window", index=frame.index, dtype="string")
    status.loc[frame["discharged_before_horizon_end"]] = "discharged_before_horizon_end"
    status.loc[
        (~frame["follow_up_image_observed"]) & frame["follow_up_study_observed"]
    ] = "follow_up_study_but_local_image_missing"
    status.loc[frame["follow_up_image_observed"]] = "follow_up_image_observed"
    return status


def build_followup_table(pairs, target_hour):
    table = pairs.copy()

    bool_columns = [
        "treated",
        "candidate_selected",
        "primary_analysis_eligible",
        "discharged_before_horizon_end",
        "follow_up_study_observed",
        "follow_up_image_observed",
        "image_exists_0",
        "image_exists_1",
        "strategy_deviation_before_horizon",
        "competing_any_before_horizon",
    ]
    for column in bool_columns:
        if column in table.columns:
            table[column] = coerce_bool(table[column])

    datetime_columns = [
        "t0",
        "t1",
        "admittime",
        "dischtime",
        "follow_up_any_time",
        "target_follow_up_time",
        "follow_up_window_start",
        "follow_up_window_end",
    ]
    for column in datetime_columns:
        if column in table.columns:
            table[column] = pd.to_datetime(table[column], errors="coerce")

    if table.duplicated(subset=["subject_id", "hadm_id"]).any():
        duplicates = int(table.duplicated(subset=["subject_id", "hadm_id"]).sum())
        raise ValueError(f"Expected one row per admission in selected cohort, found {duplicates} duplicates")

    table["follow_up_observation_label"] = table["follow_up_image_observed"].astype(int)
    table["follow_up_study_label"] = table["follow_up_study_observed"].astype(int)
    table["protocol_adherent_until_horizon"] = (~table["strategy_deviation_before_horizon"]).astype(int)
    table["observation_status"] = derive_observation_status(table)
    table["usable_primary_image_outcome"] = (
        table["follow_up_image_observed"] & (~table["strategy_deviation_before_horizon"])
    ).astype(int)
    table["follow_up_abs_error_hours"] = pd.NA
    if "hours_diff" in table.columns:
        numeric_hours = pd.to_numeric(table["hours_diff"], errors="coerce")
        table["hours_diff"] = numeric_hours
        table["follow_up_abs_error_hours"] = (numeric_hours - float(target_hour)).abs()

    table["treated_group"] = table["treated"].map({True: "treated", False: "control"}).astype("string")
    return table


def summarize_group_counts(table, key):
    counts = {}
    grouped = table.groupby(["treated_group", key], dropna=False).size()
    for (treated_group, value), count in grouped.items():
        counts.setdefault(treated_group, {})[str(value)] = int(count)
    return counts


def summarize_followup_table(table, config, pairs_path):
    numeric_followup = pd.to_numeric(table["hours_diff"], errors="coerce")
    observed = table.loc[table["follow_up_image_observed"]].copy()

    summary = {
        "study_name": config["study_name"],
        "source_pairs_path": str(pairs_path),
        "counts": {
            "selected_primary_baselines": int(len(table)),
            "treated_selected_primary_baselines": int(table["treated"].sum()),
            "control_selected_primary_baselines": int((~table["treated"]).sum()),
            "follow_up_image_observed": int(table["follow_up_image_observed"].sum()),
            "follow_up_study_observed": int(table["follow_up_study_observed"].sum()),
            "usable_primary_image_outcome": int(table["usable_primary_image_outcome"].sum()),
            "strategy_deviation_before_horizon": int(table["strategy_deviation_before_horizon"].sum()),
            "discharged_before_horizon_end": int(table["discharged_before_horizon_end"].sum()),
        },
        "rates_by_treatment": {
            "follow_up_image_observed": {
                group: float(value)
                for group, value in table.groupby("treated_group")["follow_up_image_observed"].mean().items()
            },
            "follow_up_study_observed": {
                group: float(value)
                for group, value in table.groupby("treated_group")["follow_up_study_observed"].mean().items()
            },
            "strategy_deviation_before_horizon": {
                group: float(value)
                for group, value in table.groupby("treated_group")["strategy_deviation_before_horizon"].mean().items()
            },
            "usable_primary_image_outcome": {
                group: float(value)
                for group, value in table.groupby("treated_group")["usable_primary_image_outcome"].mean().items()
            },
        },
        "counts_by_treatment": {
            "observation_status": summarize_group_counts(table, "observation_status"),
            "strategy_deviation_before_horizon": summarize_group_counts(
                table, "strategy_deviation_before_horizon"
            ),
        },
        "follow_up_timing_hours_for_observed_images": {
            "count": int(observed["follow_up_image_observed"].sum()),
            "mean": None if observed.empty else float(observed["hours_diff"].mean()),
            "median": None if observed.empty else float(observed["hours_diff"].median()),
            "min": None if observed.empty else float(observed["hours_diff"].min()),
            "max": None if observed.empty else float(observed["hours_diff"].max()),
            "mean_abs_error_from_target_hour": None
            if observed.empty
            else float(observed["follow_up_abs_error_hours"].mean()),
        },
        "cohort_risk_flags": {
            "treated_follow_up_image_observed_below_150": bool(
                int(table.loc[table["treated"], "follow_up_image_observed"].sum()) < 150
            ),
            "treated_strategy_deviation_rate_above_0_5": bool(
                table.loc[table["treated"], "strategy_deviation_before_horizon"].mean() > 0.5
            ),
            "follow_up_image_observation_gap_treated_minus_control": float(
                table.loc[table["treated"], "follow_up_image_observed"].mean()
                - table.loc[~table["treated"], "follow_up_image_observed"].mean()
            ),
        },
    }

    if numeric_followup.notna().any():
        summary["all_observed_hours_diff_available"] = int(numeric_followup.notna().sum())

    return summary


def write_outputs(output_dir, table, summary, output_manifest):
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "followup_selection_table.csv"
    summary_path = output_dir / "followup_selection_summary.json"

    table.to_csv(table_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Saved follow-up selection table to {table_path}")
    print(f"Saved follow-up selection summary to {summary_path}")
    print(
        "[summary] "
        f"selected={summary['counts']['selected_primary_baselines']} "
        f"observed_image={summary['counts']['follow_up_image_observed']} "
        f"usable_primary={summary['counts']['usable_primary_image_outcome']} "
        f"treated_observed={int(table.loc[table['treated'], 'follow_up_image_observed'].sum())}"
    )

    if output_manifest:
        manifest_path = Path(output_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "script": "build_followup_selection_table.py",
            "table_path": str(table_path),
            "summary_path": str(summary_path),
            "counts": summary["counts"],
        }
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(config)
    pairs_path = Path(args.pairs_path) if args.pairs_path else infer_pairs_path(output_dir)
    pairs = pd.read_csv(pairs_path, low_memory=False)
    target_hour = config["follow_up"]["target_hour"]

    table = build_followup_table(pairs, target_hour=target_hour)
    summary = summarize_followup_table(table, config=config, pairs_path=pairs_path)
    write_outputs(output_dir, table, summary, args.output_manifest)


if __name__ == "__main__":
    main()
