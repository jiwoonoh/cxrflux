#!/usr/bin/env python3

import argparse
import gzip
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import zlib


DIAGNOSIS_PATTERNS = {
    "heart_failure_dx": re.compile(r"heart failure|cardiomyopathy", re.IGNORECASE),
    "pulmonary_edema_dx": re.compile(r"pulmonary edema", re.IGNORECASE),
    "ckd_dx": re.compile(
        r"chronic kidney disease|end stage renal disease|hypertensive chronic kidney disease",
        re.IGNORECASE,
    ),
    "aki_dx": re.compile(r"acute kidney failure|acute renal failure|acute kidney injury", re.IGNORECASE),
    "hypertension_dx": re.compile(r"hypertension|hypertensive", re.IGNORECASE),
    "diabetes_dx": re.compile(r"diabetes mellitus|\bdiabetes\b", re.IGNORECASE),
    "cad_dx": re.compile(
        r"coronary artery disease|coronary atherosclerosis|ischemic heart disease|myocardial infarction",
        re.IGNORECASE,
    ),
    "copd_dx": re.compile(
        r"chronic obstructive pulmonary disease|emphysema|chronic bronchitis",
        re.IGNORECASE,
    ),
}

LAB_TARGETS = {
    "creatinine": {
        "allowed_labels": {"Creatinine", "Creatinine, Serum"},
        "allowed_categories": {"Chemistry"},
    },
    "urea_nitrogen": {
        "allowed_labels": {"Urea Nitrogen"},
        "allowed_categories": {"Chemistry"},
    },
    "sodium": {
        "allowed_labels": {"Sodium"},
        "allowed_categories": {"Chemistry"},
    },
    "potassium": {
        "allowed_labels": {"Potassium"},
        "allowed_categories": {"Chemistry"},
    },
    "chloride": {
        "allowed_labels": {"Chloride"},
        "allowed_categories": {"Chemistry"},
    },
    "bicarbonate": {
        "allowed_labels": {"Bicarbonate"},
        "allowed_categories": {"Chemistry"},
    },
    "ntprobnp": {
        "label_pattern": re.compile(r"ntprobnp", re.IGNORECASE),
        "allowed_categories": {"Chemistry"},
    },
    "hemoglobin": {
        "allowed_labels": {"Hemoglobin"},
        "allowed_categories": {"Hematology"},
    },
    "white_blood_cells": {
        "allowed_labels": {"White Blood Cells"},
        "allowed_categories": {"Hematology"},
    },
}

OMR_TARGETS = {
    "BMI (kg/m2)": "omr_bmi",
    "Weight (Lbs)": "omr_weight_lbs",
    "Height (Inches)": "omr_height_inches",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build pair-level pre-treatment causal covariates for the IV-furosemide cohort"
    )
    parser.add_argument(
        "--pairs-path",
        default="data_preparation/cxr_pairs_frontal_iv_furosemide_clean.csv",
    )
    parser.add_argument(
        "--admin-events-path",
        default="data_preparation/cxr_pairs_frontal_iv_furosemide_admin_events.csv",
    )
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
        "--labevents-path",
        default="physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz",
    )
    parser.add_argument(
        "--labitems-path",
        default="physionet.org/files/mimiciv/3.1/hosp/d_labitems.csv.gz",
    )
    parser.add_argument(
        "--omr-path",
        default="physionet.org/files/mimiciv/3.1/hosp/omr.csv.gz",
    )
    parser.add_argument(
        "--output-prefix",
        default="data_preparation/cxr_pairs_frontal_iv_furosemide",
    )
    parser.add_argument("--lab-lookback-hours", type=float, default=48.0)
    parser.add_argument("--omr-lookback-days", type=float, default=3650.0)
    parser.add_argument("--prior-furo-lookback-hours", type=float, default=48.0)
    parser.add_argument("--chunksize", type=int, default=250000)
    return parser.parse_args()


def normalize_string(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_id_series(series):
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()


def filter_in_chunks(path, usecols, predicate, chunksize):
    frames = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        filtered = chunk.loc[predicate(chunk)].copy()
        if not filtered.empty:
            frames.append(filtered)

    if not frames:
        return pd.DataFrame(columns=usecols)

    return pd.concat(frames, ignore_index=True)


def _read_first_gzip_member(path):
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    chunks = []

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            data = decompressor.decompress(chunk)
            if data:
                chunks.append(data)
            if decompressor.eof:
                break

    if not decompressor.eof:
        raise gzip.BadGzipFile(f"Could not decompress first gzip member from {path}")

    return b"".join(chunks)


def read_csv_resilient(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except (gzip.BadGzipFile, OSError, EOFError) as exc:
        if not str(path).endswith(".gz"):
            raise
        message = str(exc)
        if "Not a gzipped file" not in message and "Compressed file ended before" not in message:
            raise
        data = _read_first_gzip_member(path)
        return pd.read_csv(io.BytesIO(data), **kwargs)


def load_pairs(path):
    pairs = pd.read_csv(path, low_memory=False)
    pairs["subject_id"] = normalize_id_series(pairs["subject_id"])
    pairs["hadm_id"] = normalize_id_series(pairs["hadm_id"])
    pairs["study_id_0"] = normalize_id_series(pairs["study_id_0"])
    pairs["study_id_1"] = normalize_id_series(pairs["study_id_1"])
    pairs["t0"] = pd.to_datetime(pairs["t0"], errors="coerce")
    pairs["t1"] = pd.to_datetime(pairs["t1"], errors="coerce")
    pairs = pairs.dropna(subset=["subject_id", "hadm_id", "t0", "t1"]).copy()
    pairs["pair_id"] = (
        pairs["subject_id"]
        + "_"
        + pairs["hadm_id"]
        + "_"
        + pairs["study_id_0"]
        + "_"
        + pairs["study_id_1"]
    )
    pairs["treated"] = pairs["treated"].fillna(False).astype(bool)
    return pairs.reset_index(drop=True)


def latest_event_lookup(base, events, group_col, base_time_col, event_time_col, value_cols, tolerance=None):
    result = base[["pair_id"]].copy()
    result[event_time_col] = pd.NaT
    for column in value_cols:
        result[column] = pd.NA

    if events.empty:
        return result

    event_groups = {
        group_value: group_frame.sort_values(event_time_col).reset_index(drop=True)
        for group_value, group_frame in events.groupby(group_col, sort=False)
    }

    tolerance_ns = None if tolerance is None else np.timedelta64(int(tolerance.value), "ns")

    for group_value, base_group in base.groupby(group_col, sort=False):
        event_group = event_groups.get(group_value)
        if event_group is None or event_group.empty:
            continue

        event_times = event_group[event_time_col].to_numpy(dtype="datetime64[ns]")
        base_times = base_group[base_time_col].to_numpy(dtype="datetime64[ns]")
        matched_indices = event_times.searchsorted(base_times, side="right") - 1
        valid = matched_indices >= 0
        if not valid.any():
            continue

        if tolerance_ns is not None:
            deltas = base_times[valid] - event_times[matched_indices[valid]]
            valid_positions = np.where(valid)[0][deltas <= tolerance_ns]
        else:
            valid_positions = np.where(valid)[0]

        if len(valid_positions) == 0:
            continue

        base_indices = base_group.index.to_numpy()[valid_positions]
        event_rows = event_group.iloc[matched_indices[valid_positions]]

        result.loc[base_indices, event_time_col] = event_rows[event_time_col].values
        for column in value_cols:
            result.loc[base_indices, column] = event_rows[column].values

    return result


def build_demographic_features(pairs, patients_path, admissions_path):
    patients = read_csv_resilient(
        patients_path,
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        low_memory=False,
    )
    admissions = read_csv_resilient(
        admissions_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "admission_type",
            "admission_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "edregtime",
            "edouttime",
            "hospital_expire_flag",
        ],
        low_memory=False,
    )

    patients["subject_id"] = normalize_id_series(patients["subject_id"])
    admissions["subject_id"] = normalize_id_series(admissions["subject_id"])
    admissions["hadm_id"] = normalize_id_series(admissions["hadm_id"])
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    admissions["edregtime"] = pd.to_datetime(admissions["edregtime"], errors="coerce")
    admissions["edouttime"] = pd.to_datetime(admissions["edouttime"], errors="coerce")

    merged = pairs[["pair_id", "subject_id", "hadm_id", "t0"]].merge(
        patients,
        on="subject_id",
        how="left",
    )
    merged = merged.merge(
        admissions,
        on=["subject_id", "hadm_id"],
        how="left",
    )

    merged["age_at_t0"] = merged["anchor_age"] + (merged["t0"].dt.year - merged["anchor_year"])
    merged["hours_since_admission"] = (
        (merged["t0"] - merged["admittime"]).dt.total_seconds() / 3600.0
    )
    merged["hours_since_ed_registration"] = (
        (merged["t0"] - merged["edregtime"]).dt.total_seconds() / 3600.0
    )
    merged["ed_before_t0"] = (
        merged["edregtime"].notna() & (merged["edregtime"] <= merged["t0"])
    ).astype("int8")
    merged["baseline_after_admission"] = (
        merged["admittime"].notna() & (merged["t0"] >= merged["admittime"])
    ).astype("int8")

    return merged[
        [
            "pair_id",
            "gender",
            "age_at_t0",
            "admission_type",
            "admission_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "hospital_expire_flag",
            "hours_since_admission",
            "hours_since_ed_registration",
            "ed_before_t0",
            "baseline_after_admission",
        ]
    ]


def build_service_features(pairs, services_path):
    services = read_csv_resilient(
        services_path,
        usecols=["hadm_id", "transfertime", "curr_service"],
        low_memory=False,
    )
    services["hadm_id"] = normalize_id_series(services["hadm_id"])
    services["transfertime"] = pd.to_datetime(services["transfertime"], errors="coerce")
    services = services.dropna(subset=["hadm_id", "transfertime"]).copy()
    services = services[services["hadm_id"].isin(set(pairs["hadm_id"]))].copy()

    left = pairs[["pair_id", "hadm_id", "t0"]].copy()
    latest = latest_event_lookup(
        left,
        services[["hadm_id", "transfertime", "curr_service"]].copy(),
        group_col="hadm_id",
        base_time_col="t0",
        event_time_col="transfertime",
        value_cols=["curr_service"],
    )

    transition_counts = left.merge(services, on="hadm_id", how="left")
    transition_counts = transition_counts[
        transition_counts["transfertime"].notna()
        & (transition_counts["transfertime"] <= transition_counts["t0"])
    ]
    transition_counts = (
        transition_counts.groupby("pair_id").size().rename("service_transfers_before_t0").reset_index()
    )

    features = latest[["pair_id", "curr_service"]].rename(columns={"curr_service": "service_at_t0"})
    features = features.merge(transition_counts, on="pair_id", how="left")
    features["service_transfers_before_t0"] = (
        features["service_transfers_before_t0"].fillna(0).astype("int16")
    )
    return features


def build_icu_features(pairs, icustays_path):
    icustays = read_csv_resilient(
        icustays_path,
        usecols=["hadm_id", "stay_id", "first_careunit", "intime", "outtime"],
        low_memory=False,
    )
    icustays["hadm_id"] = normalize_id_series(icustays["hadm_id"])
    icustays["intime"] = pd.to_datetime(icustays["intime"], errors="coerce")
    icustays["outtime"] = pd.to_datetime(icustays["outtime"], errors="coerce")
    icustays = icustays.dropna(subset=["hadm_id", "intime", "outtime"]).copy()
    icustays = icustays[icustays["hadm_id"].isin(set(pairs["hadm_id"]))].copy()

    merged = pairs[["pair_id", "hadm_id", "t0"]].merge(icustays, on="hadm_id", how="left")
    valid = merged["intime"].notna()
    before_mask = valid & (merged["intime"] < merged["t0"])
    active_mask = valid & (merged["intime"] <= merged["t0"]) & (merged["t0"] < merged["outtime"])

    overlap_start = merged["intime"]
    overlap_end = merged[["outtime", "t0"]].min(axis=1)
    merged["icu_hours_before_t0_component"] = (
        (overlap_end - overlap_start).dt.total_seconds() / 3600.0
    ).clip(lower=0).fillna(0)
    merged.loc[~before_mask, "icu_hours_before_t0_component"] = 0

    features = pairs[["pair_id"]].copy()
    before_any = before_mask.groupby(merged["pair_id"]).any().astype("int8").rename("any_icu_before_t0")
    active_any = active_mask.groupby(merged["pair_id"]).any().astype("int8").rename("in_icu_at_t0")
    stay_counts = before_mask.groupby(merged["pair_id"]).sum().astype("int16").rename("icu_stays_before_t0")
    hour_sums = (
        merged.groupby("pair_id")["icu_hours_before_t0_component"].sum().astype("float32").rename("icu_hours_before_t0")
    )

    for series in [before_any, active_any, stay_counts, hour_sums]:
        features = features.merge(series.reset_index(), on="pair_id", how="left")

    active_rows = merged.loc[active_mask, ["pair_id", "intime", "first_careunit"]].copy()
    active_rows = active_rows.sort_values(["pair_id", "intime"])
    active_rows = active_rows.groupby("pair_id", as_index=False).tail(1)
    active_rows = active_rows.rename(columns={"first_careunit": "icu_careunit_at_t0"})

    features = features.merge(active_rows[["pair_id", "icu_careunit_at_t0"]], on="pair_id", how="left")
    features["any_icu_before_t0"] = features["any_icu_before_t0"].fillna(0).astype("int8")
    features["in_icu_at_t0"] = features["in_icu_at_t0"].fillna(0).astype("int8")
    features["icu_stays_before_t0"] = features["icu_stays_before_t0"].fillna(0).astype("int16")
    features["icu_hours_before_t0"] = features["icu_hours_before_t0"].fillna(0).astype("float32")
    return features


def build_diagnosis_features(pairs, diagnoses_path, diagnosis_titles_path, chunksize):
    hadm_ids = set(pairs["hadm_id"])
    diagnoses = filter_in_chunks(
        diagnoses_path,
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        predicate=lambda chunk: normalize_id_series(chunk["hadm_id"]).isin(hadm_ids),
        chunksize=chunksize,
    )
    if diagnoses.empty:
        features = pairs[["pair_id", "hadm_id"]].copy()
        features["diagnosis_count"] = 0
        for name in DIAGNOSIS_PATTERNS:
            features[f"has_{name}"] = 0
        return features.drop(columns=["hadm_id"])

    diagnoses["hadm_id"] = normalize_id_series(diagnoses["hadm_id"])
    diagnoses["icd_code"] = diagnoses["icd_code"].astype("string").str.strip()
    diagnoses["icd_version"] = diagnoses["icd_version"].astype("string").str.strip()

    titles = read_csv_resilient(
        diagnosis_titles_path,
        usecols=["icd_code", "icd_version", "long_title"],
        low_memory=False,
    )
    titles["icd_code"] = titles["icd_code"].astype("string").str.strip()
    titles["icd_version"] = titles["icd_version"].astype("string").str.strip()
    diagnoses = diagnoses.merge(titles, on=["icd_code", "icd_version"], how="left")
    diagnoses["long_title"] = diagnoses["long_title"].astype("string").fillna("")

    aggregated = (
        diagnoses.groupby("hadm_id")
        .agg(
            diagnosis_count=("icd_code", "count"),
            diagnosis_titles=("long_title", lambda s: " | ".join(sorted(set(v for v in s if normalize_string(v))))),
        )
        .reset_index()
    )

    for pattern_name, pattern in DIAGNOSIS_PATTERNS.items():
        aggregated[f"has_{pattern_name}"] = aggregated["diagnosis_titles"].str.contains(pattern, na=False).astype("int8")

    features = pairs[["pair_id", "hadm_id"]].merge(aggregated, on="hadm_id", how="left")
    features["diagnosis_count"] = features["diagnosis_count"].fillna(0).astype("int16")
    for pattern_name in DIAGNOSIS_PATTERNS:
        column = f"has_{pattern_name}"
        features[column] = features[column].fillna(0).astype("int8")

    return features.drop(columns=["hadm_id", "diagnosis_titles"])


def discover_lab_itemids(labitems_path):
    labitems = read_csv_resilient(
        labitems_path,
        usecols=["itemid", "label", "fluid", "category"],
        low_memory=False,
    )

    itemid_to_lab = {}
    used_itemids = {}
    for lab_name, config in LAB_TARGETS.items():
        mask = pd.Series(True, index=labitems.index)
        if "allowed_labels" in config:
            mask &= labitems["label"].isin(config["allowed_labels"])
        if "label_pattern" in config:
            mask &= labitems["label"].astype("string").str.contains(config["label_pattern"], na=False)
        if "allowed_categories" in config:
            mask &= labitems["category"].isin(config["allowed_categories"])
        selected = labitems.loc[mask, "itemid"].astype(int).tolist()
        used_itemids[lab_name] = selected
        for itemid in selected:
            itemid_to_lab[itemid] = lab_name

    return itemid_to_lab, used_itemids


def load_target_labevents(labevents_path, pairs, itemid_to_lab, lookback_hours, chunksize):
    if not itemid_to_lab:
        return pd.DataFrame(columns=["hadm_id", "charttime", "valuenum", "lab_name"])

    hadm_ids = set(pairs["hadm_id"])
    target_itemids = set(itemid_to_lab)
    min_time = pairs["t0"].min() - pd.to_timedelta(lookback_hours, unit="h")
    max_time = pairs["t0"].max()

    events = filter_in_chunks(
        labevents_path,
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        predicate=lambda chunk: normalize_id_series(chunk["hadm_id"]).isin(hadm_ids)
        & chunk["itemid"].isin(target_itemids),
        chunksize=chunksize,
    )

    if events.empty:
        return pd.DataFrame(columns=["hadm_id", "charttime", "valuenum", "lab_name"])

    events["hadm_id"] = normalize_id_series(events["hadm_id"])
    events["charttime"] = pd.to_datetime(events["charttime"], errors="coerce")
    events["valuenum"] = pd.to_numeric(events["valuenum"], errors="coerce")
    events["lab_name"] = events["itemid"].map(itemid_to_lab)
    events = events.dropna(subset=["hadm_id", "charttime", "valuenum", "lab_name"]).copy()
    events = events[(events["charttime"] >= min_time) & (events["charttime"] <= max_time)].copy()
    return events.reset_index(drop=True)


def build_lab_features(pairs, events, lookback_hours):
    lookback = pd.to_timedelta(lookback_hours, unit="h")
    base = pairs[["pair_id", "hadm_id", "t0"]].copy()
    features = base[["pair_id"]].copy()

    for lab_name in LAB_TARGETS:
        lab_events = events[events["lab_name"] == lab_name].copy()
        value_column = f"lab_{lab_name}"
        age_column = f"lab_{lab_name}_hours_before_t0"

        if lab_events.empty:
            features[value_column] = pd.NA
            features[age_column] = pd.NA
            continue

        latest = latest_event_lookup(
            base,
            lab_events[["hadm_id", "charttime", "valuenum"]].copy(),
            group_col="hadm_id",
            base_time_col="t0",
            event_time_col="charttime",
            value_cols=["valuenum"],
            tolerance=lookback,
        )
        chart_times = pd.to_datetime(latest["charttime"], errors="coerce")
        features[value_column] = latest["valuenum"].values
        features[age_column] = ((base["t0"] - chart_times).dt.total_seconds() / 3600.0).values

    return features


def parse_omr_numeric(row):
    result_name = normalize_string(row["result_name"])
    value = normalize_string(row["result_value"])

    if not value:
        return pd.NA, pd.NA, pd.NA

    if result_name == "Blood Pressure":
        match = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", value)
        if not match:
            return pd.NA, pd.NA, pd.NA
        return pd.NA, float(match.group(1)), float(match.group(2))

    try:
        return float(value), pd.NA, pd.NA
    except ValueError:
        return pd.NA, pd.NA, pd.NA


def build_omr_features(pairs, omr_path, lookback_days, chunksize):
    subject_ids = set(pairs["subject_id"])
    target_names = set(OMR_TARGETS) | {"Blood Pressure"}

    omr = filter_in_chunks(
        omr_path,
        usecols=["subject_id", "chartdate", "result_name", "result_value"],
        predicate=lambda chunk: normalize_id_series(chunk["subject_id"]).isin(subject_ids)
        & chunk["result_name"].isin(target_names),
        chunksize=chunksize,
    )
    if omr.empty:
        return pairs[["pair_id"]].copy()

    omr["subject_id"] = normalize_id_series(omr["subject_id"])
    omr["chartdate"] = pd.to_datetime(omr["chartdate"], errors="coerce")
    parsed = omr.apply(parse_omr_numeric, axis=1, result_type="expand")
    parsed.columns = ["numeric_value", "bp_systolic", "bp_diastolic"]
    omr = pd.concat([omr, parsed], axis=1)
    omr = omr.dropna(subset=["subject_id", "chartdate"]).copy()

    lookback = pd.to_timedelta(lookback_days, unit="d")
    base = pairs[["pair_id", "subject_id", "t0"]].copy()
    features = base[["pair_id"]].copy()

    for result_name, prefix in OMR_TARGETS.items():
        result_rows = omr[omr["result_name"] == result_name].copy()
        value_column = prefix
        age_column = f"{prefix}_days_before_t0"

        if result_rows.empty:
            features[value_column] = pd.NA
            features[age_column] = pd.NA
            continue

        latest = latest_event_lookup(
            base,
            result_rows[["subject_id", "chartdate", "numeric_value"]].copy(),
            group_col="subject_id",
            base_time_col="t0",
            event_time_col="chartdate",
            value_cols=["numeric_value"],
            tolerance=lookback,
        )
        chart_dates = pd.to_datetime(latest["chartdate"], errors="coerce")
        features[value_column] = latest["numeric_value"].values
        features[age_column] = ((base["t0"] - chart_dates).dt.total_seconds() / 86400.0).values

    bp_rows = omr[omr["result_name"] == "Blood Pressure"].copy()
    if bp_rows.empty:
        features["omr_bp_systolic"] = pd.NA
        features["omr_bp_diastolic"] = pd.NA
        features["omr_bp_days_before_t0"] = pd.NA
    else:
        latest = latest_event_lookup(
            base,
            bp_rows[["subject_id", "chartdate", "bp_systolic", "bp_diastolic"]].copy(),
            group_col="subject_id",
            base_time_col="t0",
            event_time_col="chartdate",
            value_cols=["bp_systolic", "bp_diastolic"],
            tolerance=lookback,
        )
        chart_dates = pd.to_datetime(latest["chartdate"], errors="coerce")
        features["omr_bp_systolic"] = latest["bp_systolic"].values
        features["omr_bp_diastolic"] = latest["bp_diastolic"].values
        features["omr_bp_days_before_t0"] = (
            (base["t0"] - chart_dates).dt.total_seconds() / 86400.0
        ).values

    return features


def build_prior_furo_features(pairs, admin_events_path, lookback_hours):
    admins = pd.read_csv(
        admin_events_path,
        usecols=["hadm_id", "event_is_positive", "route_label", "treatment_event_time"],
        low_memory=False,
    )
    admins["hadm_id"] = normalize_id_series(admins["hadm_id"])
    admins["treatment_event_time"] = pd.to_datetime(admins["treatment_event_time"], errors="coerce")
    admins["event_is_positive"] = admins["event_is_positive"].fillna(False).astype(bool)
    admins["route_label"] = admins["route_label"].astype("string").fillna("")
    admins = admins[
        admins["event_is_positive"]
        & admins["hadm_id"].isin(set(pairs["hadm_id"]))
        & admins["treatment_event_time"].notna()
    ].copy()

    merged = pairs[["pair_id", "hadm_id", "t0"]].merge(admins, on="hadm_id", how="left")
    merged = merged[merged["treatment_event_time"].notna() & (merged["treatment_event_time"] < merged["t0"])].copy()
    merged["hours_before_t0"] = (
        (merged["t0"] - merged["treatment_event_time"]).dt.total_seconds() / 3600.0
    )

    features = pairs[["pair_id"]].copy()
    if merged.empty:
        for column in [
            "prior_iv_furo_count_before_t0",
            "prior_oral_furo_count_before_t0",
            "prior_ambiguous_furo_count_before_t0",
            "prior_iv_furo_count_lookback",
            "prior_oral_furo_count_lookback",
        ]:
            features[column] = 0
        return features

    for route_label in ["iv", "oral", "ambiguous"]:
        counts = (
            merged.loc[merged["route_label"] == route_label]
            .groupby("pair_id")
            .size()
            .rename(f"prior_{route_label}_furo_count_before_t0")
            .reset_index()
        )
        features = features.merge(counts, on="pair_id", how="left")

    recent = merged[merged["hours_before_t0"] <= lookback_hours].copy()
    for route_label in ["iv", "oral"]:
        counts = (
            recent.loc[recent["route_label"] == route_label]
            .groupby("pair_id")
            .size()
            .rename(f"prior_{route_label}_furo_count_lookback")
            .reset_index()
        )
        features = features.merge(counts, on="pair_id", how="left")

    count_columns = [
        "prior_iv_furo_count_before_t0",
        "prior_oral_furo_count_before_t0",
        "prior_ambiguous_furo_count_before_t0",
        "prior_iv_furo_count_lookback",
        "prior_oral_furo_count_lookback",
    ]
    for column in count_columns:
        if column not in features:
            features[column] = 0
        features[column] = features[column].fillna(0).astype("int16")

    return features


def build_summary(causal_ready, covariates, used_itemids):
    numeric_feature_columns = [
        column
        for column in covariates.columns
        if column not in {"pair_id"}
    ]
    missingness = {
        column: float(covariates[column].isna().mean())
        for column in numeric_feature_columns
    }

    return {
        "n_pairs": int(len(causal_ready)),
        "n_subjects": int(causal_ready["subject_id"].nunique()),
        "n_hadm_ids": int(causal_ready["hadm_id"].nunique()),
        "treated_pairs": int(causal_ready["treated"].sum()),
        "control_pairs": int((~causal_ready["treated"]).sum()),
        "lab_itemids": {name: values for name, values in used_itemids.items() if values},
        "feature_missingness": missingness,
    }


def main():
    args = parse_args()

    pairs = load_pairs(args.pairs_path)
    demographic_features = build_demographic_features(
        pairs,
        args.patients_path,
        args.admissions_path,
    )
    service_features = build_service_features(pairs, args.services_path)
    icu_features = build_icu_features(pairs, args.icustays_path)
    diagnosis_features = build_diagnosis_features(
        pairs,
        args.diagnoses_path,
        args.diagnosis_titles_path,
        args.chunksize,
    )
    prior_furo_features = build_prior_furo_features(
        pairs,
        args.admin_events_path,
        args.prior_furo_lookback_hours,
    )

    itemid_to_lab, used_itemids = discover_lab_itemids(args.labitems_path)
    labevents = load_target_labevents(
        args.labevents_path,
        pairs,
        itemid_to_lab,
        args.lab_lookback_hours,
        args.chunksize,
    )
    lab_features = build_lab_features(pairs, labevents, args.lab_lookback_hours)
    omr_features = build_omr_features(
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
    summary = build_summary(causal_ready, covariates, used_itemids)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    covariates_path = Path(f"{args.output_prefix}_causal_covariates.csv")
    causal_ready_path = Path(f"{args.output_prefix}_causal_ready.csv")
    summary_path = Path(f"{args.output_prefix}_causal_covariates_summary.json")

    covariates.to_csv(covariates_path, index=False)
    causal_ready.to_csv(causal_ready_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved covariates to {covariates_path}")
    print(f"Saved causal-ready pairs to {causal_ready_path}")
    print(f"Saved summary to {summary_path}")
    print(
        "[summary] "
        f"pairs={summary['n_pairs']} treated={summary['treated_pairs']} "
        f"controls={summary['control_pairs']}"
    )


if __name__ == "__main__":
    main()
