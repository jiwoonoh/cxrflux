#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import pandas as pd


FRONTAL_VIEWS = {"PA", "AP"}
FUROSEMIDE_PATTERN = re.compile(r"(?:furosemide|lasix)", re.IGNORECASE)
POSITIVE_EVENT_TXT = {"Administered", "Confirmed", "Started"}
IV_ROUTE_VALUES = {"IV", "IV DRIP", "IVP", "IVPB", "INTRAVENOUS"}
ORAL_ROUTE_VALUES = {
    "PO",
    "PO/NG",
    "PO/GTUBE",
    "PO/GT",
    "PO/JTUBE",
    "NG",
    "GTUBE",
    "JTUBE",
}
HEART_FAILURE_PATTERN = re.compile(r"heart failure|congestive heart failure|cardiomyopathy", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build frontal longitudinal CXR pairs with audited IV furosemide labels"
    )
    parser.add_argument(
        "--metadata-path",
        default="mimic-cxr-2.0.0-metadata.csv.gz",
    )
    parser.add_argument(
        "--admissions-path",
        default="physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz",
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
        "--emar-path",
        default="physionet.org/files/mimiciv/3.1/hosp/emar.csv.gz",
    )
    parser.add_argument(
        "--emar-detail-path",
        default="physionet.org/files/mimiciv/3.1/hosp/emar_detail.csv.gz",
    )
    parser.add_argument(
        "--pharmacy-path",
        default="physionet.org/files/mimiciv/3.1/hosp/pharmacy.csv.gz",
    )
    parser.add_argument(
        "--prescriptions-path",
        default="physionet.org/files/mimiciv/3.1/hosp/prescriptions.csv.gz",
    )
    parser.add_argument(
        "--poe-detail-path",
        default="physionet.org/files/mimiciv/3.1/hosp/poe_detail.csv.gz",
    )
    parser.add_argument("--image-root", default="./mimic-cxr-jpg/")
    parser.add_argument("--output-prefix", default="data_preparation/cxr_pairs_frontal_iv_furosemide")
    parser.add_argument("--cohort", choices=["all", "heart_failure"], default="all")
    parser.add_argument("--min-hours", type=float, default=6.0)
    parser.add_argument("--max-hours", type=float, default=48.0)
    parser.add_argument("--admission-grace-hours", type=float, default=6.0)
    parser.add_argument("--washout-hours", type=float, default=12.0)
    parser.add_argument("--chunksize", type=int, default=250000)
    return parser.parse_args()


def normalize_string(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_id_series(series):
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()


def unique_join(series):
    values = sorted(
        {
            normalize_string(value)
            for value in series
            if normalize_string(value)
        }
    )
    return "; ".join(values)


def parse_study_timestamp(study_date, study_time):
    date_part = normalize_string(study_date).zfill(8)
    time_raw = normalize_string(study_time)

    if not date_part or not time_raw:
        return pd.NaT

    if "e" in time_raw.lower():
        time_raw = format(float(time_raw), ".6f")

    digits, _, fraction = time_raw.partition(".")
    digits = re.sub(r"[^0-9]", "", digits).zfill(6)[-6:]
    fraction = re.sub(r"[^0-9]", "", fraction)[:6].rstrip("0")

    formatted = f"{digits[:2]}:{digits[2:4]}:{digits[4:6]}"
    if fraction:
        formatted = f"{formatted}.{fraction}"

    return pd.to_datetime(f"{date_part} {formatted}", errors="coerce")


def filter_in_chunks(path, usecols, predicate, chunksize):
    frames = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        mask = predicate(chunk)
        filtered = chunk.loc[mask].copy()
        if not filtered.empty:
            frames.append(filtered)

    if not frames:
        return pd.DataFrame(columns=usecols)

    return pd.concat(frames, ignore_index=True)


def assign_hadm_ids(frame, admissions, timestamp_col, grace_hours):
    result = frame.copy()
    grace = pd.to_timedelta(grace_hours, unit="h")

    admissions = admissions.copy()
    admissions["window_start"] = admissions["admittime"] - grace
    admissions["window_end"] = admissions["dischtime"] + grace

    grouped_admissions = {
        subject_id: subject_frame.sort_values("window_start")
        for subject_id, subject_frame in admissions.groupby("subject_id", sort=False)
    }

    assigned_hadm = []
    assigned_status = []

    for row in result.itertuples(index=False):
        subject_id = normalize_string(getattr(row, "subject_id"))
        timestamp = getattr(row, timestamp_col)

        if pd.isna(timestamp) or subject_id not in grouped_admissions:
            assigned_hadm.append(pd.NA)
            assigned_status.append("unmatched")
            continue

        subject_admissions = grouped_admissions[subject_id]
        matches = subject_admissions.loc[
            (subject_admissions["window_start"] <= timestamp)
            & (timestamp <= subject_admissions["window_end"]),
            "hadm_id",
        ].astype("string")

        unique_matches = matches.dropna().astype(str).unique()
        if len(unique_matches) == 1:
            assigned_hadm.append(unique_matches[0])
            assigned_status.append("matched")
        elif len(unique_matches) == 0:
            assigned_hadm.append(pd.NA)
            assigned_status.append("unmatched")
        else:
            assigned_hadm.append(pd.NA)
            assigned_status.append("ambiguous")

    result["hadm_id"] = pd.Series(assigned_hadm, dtype="string")
    result[f"{timestamp_col}_hadm_status"] = assigned_status
    return result


def load_admissions(path):
    admissions = pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        low_memory=False,
    )
    admissions["subject_id"] = normalize_id_series(admissions["subject_id"])
    admissions["hadm_id"] = normalize_id_series(admissions["hadm_id"])
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    admissions = admissions.dropna(subset=["subject_id", "hadm_id", "admittime", "dischtime"])
    return admissions.reset_index(drop=True)


def load_frontal_studies(metadata_path, admissions, grace_hours):
    metadata = pd.read_csv(
        metadata_path,
        usecols=["dicom_id", "subject_id", "study_id", "ViewPosition", "StudyDate", "StudyTime"],
        low_memory=False,
    )
    metadata["ViewPosition"] = metadata["ViewPosition"].astype("string")
    metadata = metadata[metadata["ViewPosition"].isin(FRONTAL_VIEWS)].copy()
    metadata["subject_id"] = normalize_id_series(metadata["subject_id"])
    metadata["study_id"] = normalize_id_series(metadata["study_id"])
    metadata["dicom_id"] = metadata["dicom_id"].astype("string").str.strip()
    metadata["study_time"] = metadata.apply(
        lambda row: parse_study_timestamp(row["StudyDate"], row["StudyTime"]),
        axis=1,
    )
    metadata = metadata.dropna(subset=["subject_id", "study_id", "dicom_id", "study_time"])
    metadata["view_rank"] = metadata["ViewPosition"].map({"PA": 0, "AP": 1}).fillna(99)
    metadata = metadata.sort_values(
        ["subject_id", "study_id", "view_rank", "dicom_id"],
        kind="stable",
    )

    studies = (
        metadata.drop_duplicates(subset=["subject_id", "study_id"], keep="first")
        .loc[:, ["subject_id", "study_id", "dicom_id", "ViewPosition", "study_time"]]
        .rename(columns={"dicom_id": "cxr_id", "ViewPosition": "view_position"})
        .reset_index(drop=True)
    )

    studies = assign_hadm_ids(studies, admissions, "study_time", grace_hours)
    studies = studies[studies["study_time_hadm_status"] == "matched"].copy()
    return studies.reset_index(drop=True)


def load_heart_failure_hadm_ids(diagnoses_path, diagnosis_titles_path, chunksize):
    diagnosis_titles = pd.read_csv(
        diagnosis_titles_path,
        usecols=["icd_code", "icd_version", "long_title"],
        low_memory=False,
    )
    diagnosis_titles["icd_code"] = diagnosis_titles["icd_code"].astype("string").str.strip()
    diagnosis_titles["icd_version"] = diagnosis_titles["icd_version"].astype("string").str.strip()
    matched_titles = diagnosis_titles[
        diagnosis_titles["long_title"].astype("string").str.contains(HEART_FAILURE_PATTERN, na=False)
    ].copy()

    if matched_titles.empty:
        return set()

    matched_codes = set(
        zip(
            matched_titles["icd_code"].astype(str),
            matched_titles["icd_version"].astype(str),
        )
    )

    heart_failure_hadm_ids = set()
    for chunk in pd.read_csv(
        diagnoses_path,
        usecols=["hadm_id", "icd_code", "icd_version"],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["hadm_id"] = normalize_id_series(chunk["hadm_id"])
        chunk["icd_code"] = chunk["icd_code"].astype("string").str.strip()
        chunk["icd_version"] = chunk["icd_version"].astype("string").str.strip()
        code_pairs = list(zip(chunk["icd_code"].astype(str), chunk["icd_version"].astype(str)))
        mask = pd.Series([pair in matched_codes for pair in code_pairs], index=chunk.index)
        if mask.any():
            heart_failure_hadm_ids.update(
                chunk.loc[mask, "hadm_id"].dropna().astype(str).tolist()
            )
    return heart_failure_hadm_ids


def filter_studies_for_cohort(studies, args):
    if args.cohort == "all":
        return studies

    if args.cohort == "heart_failure":
        heart_failure_hadm_ids = load_heart_failure_hadm_ids(
            args.diagnoses_path,
            args.diagnosis_titles_path,
            args.chunksize,
        )
        return studies[studies["hadm_id"].isin(heart_failure_hadm_ids)].copy()

    raise ValueError(f"Unsupported cohort: {args.cohort}")


def aggregate_detail_rows(detail):
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "emar_id",
                "detail_routes",
                "detail_product_codes",
                "detail_product_descriptions",
                "detail_product_units",
                "detail_administration_types",
            ]
        )

    return (
        detail.groupby("emar_id", dropna=False)
        .agg(
            detail_routes=("route", unique_join),
            detail_product_codes=("product_code", unique_join),
            detail_product_descriptions=(
                "product_description",
                unique_join,
            ),
            detail_product_descriptions_other=(
                "product_description_other",
                unique_join,
            ),
            detail_product_units=("product_unit", unique_join),
            detail_administration_types=("administration_type", unique_join),
        )
        .reset_index()
    )


def aggregate_by_key(frame, key, column_map):
    if frame.empty:
        return pd.DataFrame(columns=[key, *column_map.values()])

    aggregated = (
        frame.groupby(key, dropna=False)
        .agg({source_column: unique_join for source_column in column_map})
        .reset_index()
    )
    return aggregated.rename(columns=column_map)


def load_furosemide_medication_tables(args, admissions):
    emar = filter_in_chunks(
        args.emar_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "emar_id",
            "poe_id",
            "pharmacy_id",
            "charttime",
            "medication",
            "event_txt",
        ],
        predicate=lambda chunk: chunk["medication"].astype("string").str.contains(
            FUROSEMIDE_PATTERN,
            na=False,
        ),
        chunksize=args.chunksize,
    )
    emar["subject_id"] = normalize_id_series(emar["subject_id"])
    emar["hadm_id"] = normalize_id_series(emar["hadm_id"])
    emar["emar_id"] = emar["emar_id"].astype("string").str.strip()
    emar["poe_id"] = emar["poe_id"].astype("string").str.strip()
    emar["pharmacy_id"] = emar["pharmacy_id"].astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    emar["charttime"] = pd.to_datetime(emar["charttime"], errors="coerce")
    emar = emar.dropna(subset=["subject_id", "emar_id", "charttime"])

    missing_hadm_mask = emar["hadm_id"].isna() | (emar["hadm_id"] == "")
    if missing_hadm_mask.any():
        repaired = assign_hadm_ids(
            emar.loc[missing_hadm_mask, ["subject_id", "emar_id", "charttime"]].rename(
                columns={"charttime": "event_time"}
            ),
            admissions,
            "event_time",
            args.admission_grace_hours,
        )
        emar.loc[missing_hadm_mask, "hadm_id"] = repaired["hadm_id"].values

    emar = emar.dropna(subset=["hadm_id"]).copy()

    emar_ids = set(emar["emar_id"].dropna().astype(str))
    poe_ids = set(emar["poe_id"].dropna().astype(str)) - {"", "<NA>", "nan"}
    pharmacy_ids = set(emar["pharmacy_id"].dropna().astype(str)) - {"", "<NA>", "nan"}

    detail = filter_in_chunks(
        args.emar_detail_path,
        usecols=[
            "emar_id",
            "route",
            "product_code",
            "product_description",
            "product_description_other",
            "product_unit",
            "administration_type",
        ],
        predicate=lambda chunk: chunk["emar_id"].astype("string").isin(emar_ids),
        chunksize=args.chunksize,
    )
    detail["emar_id"] = detail["emar_id"].astype("string").str.strip()
    detail_agg = aggregate_detail_rows(detail)

    pharmacy = filter_in_chunks(
        args.pharmacy_path,
        usecols=["pharmacy_id", "poe_id", "route", "medication"],
        predicate=lambda chunk: chunk["medication"].astype("string").str.contains(
            FUROSEMIDE_PATTERN,
            na=False,
        )
        | chunk["pharmacy_id"].astype("string").isin(pharmacy_ids)
        | chunk["poe_id"].astype("string").isin(poe_ids),
        chunksize=args.chunksize,
    )
    pharmacy["pharmacy_id"] = pharmacy["pharmacy_id"].astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    pharmacy["poe_id"] = pharmacy["poe_id"].astype("string").str.strip()
    pharmacy_agg = aggregate_by_key(
        pharmacy,
        "pharmacy_id",
        {
            "route": "pharmacy_routes",
            "medication": "pharmacy_medications",
            "poe_id": "pharmacy_poe_ids",
        },
    )
    pharmacy_poe_agg = aggregate_by_key(
        pharmacy,
        "poe_id",
        {
            "route": "pharmacy_routes_by_poe",
            "medication": "pharmacy_medications_by_poe",
        },
    )

    prescriptions = filter_in_chunks(
        args.prescriptions_path,
        usecols=[
            "pharmacy_id",
            "poe_id",
            "route",
            "drug",
            "formulary_drug_cd",
            "prod_strength",
            "form_unit_disp",
        ],
        predicate=lambda chunk: chunk["drug"].astype("string").str.contains(
            FUROSEMIDE_PATTERN,
            na=False,
        )
        | chunk["pharmacy_id"].astype("string").isin(pharmacy_ids)
        | chunk["poe_id"].astype("string").isin(poe_ids),
        chunksize=args.chunksize,
    )
    prescriptions["pharmacy_id"] = prescriptions["pharmacy_id"].astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    prescriptions["poe_id"] = prescriptions["poe_id"].astype("string").str.strip()
    prescriptions_agg = aggregate_by_key(
        prescriptions,
        "pharmacy_id",
        {
            "route": "prescription_routes",
            "drug": "prescription_drugs",
            "formulary_drug_cd": "prescription_codes",
            "prod_strength": "prescription_strengths",
            "form_unit_disp": "prescription_form_units",
            "poe_id": "prescription_poe_ids",
        },
    )
    prescriptions_poe_agg = aggregate_by_key(
        prescriptions,
        "poe_id",
        {
            "route": "prescription_routes_by_poe",
            "drug": "prescription_drugs_by_poe",
            "formulary_drug_cd": "prescription_codes_by_poe",
            "prod_strength": "prescription_strengths_by_poe",
            "form_unit_disp": "prescription_form_units_by_poe",
        },
    )

    poe_detail = filter_in_chunks(
        args.poe_detail_path,
        usecols=["poe_id", "field_name", "field_value"],
        predicate=lambda chunk: chunk["poe_id"].astype("string").isin(poe_ids)
        & chunk["field_name"].astype("string").str.contains("route", case=False, na=False),
        chunksize=args.chunksize,
    )
    poe_detail["poe_id"] = poe_detail["poe_id"].astype("string").str.strip()
    poe_detail_agg = aggregate_by_key(
        poe_detail,
        "poe_id",
        {"field_value": "poe_routes"},
    )

    admins = emar.merge(detail_agg, on="emar_id", how="left")
    admins = admins.merge(pharmacy_agg, on="pharmacy_id", how="left")
    admins = admins.merge(prescriptions_agg, on="pharmacy_id", how="left")
    admins = admins.merge(pharmacy_poe_agg, on="poe_id", how="left")
    admins = admins.merge(prescriptions_poe_agg, on="poe_id", how="left")
    admins = admins.merge(poe_detail_agg, on="poe_id", how="left")

    return admins


def split_unique_values(*values):
    parts = set()
    for value in values:
        if pd.isna(value):
            continue
        for token in str(value).split(";"):
            token = token.strip()
            if token:
                parts.add(token)
    return parts


def is_iv_route(route):
    route = normalize_string(route).upper()
    return route in IV_ROUTE_VALUES or route.startswith("IV")


def is_oral_route(route):
    route = normalize_string(route).upper()
    return route in ORAL_ROUTE_VALUES or route.startswith("PO")


def classify_route(row):
    route_values = split_unique_values(
        row.get("detail_routes", ""),
        row.get("pharmacy_routes", ""),
        row.get("prescription_routes", ""),
        row.get("pharmacy_routes_by_poe", ""),
        row.get("prescription_routes_by_poe", ""),
        row.get("poe_routes", ""),
    )
    explicit_iv = sorted(route for route in route_values if is_iv_route(route))
    explicit_oral = sorted(route for route in route_values if is_oral_route(route))

    product_tokens = " | ".join(
        value
        for value in [
            row.get("detail_product_codes", ""),
            row.get("detail_product_descriptions", ""),
            row.get("detail_product_descriptions_other", ""),
            row.get("detail_product_units", ""),
            row.get("prescription_codes", ""),
            row.get("prescription_strengths", ""),
            row.get("prescription_form_units", ""),
            row.get("pharmacy_medications_by_poe", ""),
            row.get("prescription_drugs_by_poe", ""),
            row.get("prescription_codes_by_poe", ""),
            row.get("prescription_strengths_by_poe", ""),
            row.get("prescription_form_units_by_poe", ""),
            row.get("pharmacy_medications", ""),
            row.get("prescription_drugs", ""),
        ]
        if normalize_string(value)
    ).lower()

    iv_hint = any(
        token in product_tokens
        for token in [
            "furo40i",
            "furo20i",
            "furo80i",
            "vial",
            "inj",
            "inject",
            "mg/ml",
            "mg / ml",
        ]
    )
    unit_tokens = {token.upper() for token in split_unique_values(
        row.get("detail_product_units", ""),
        row.get("prescription_form_units", ""),
        row.get("prescription_form_units_by_poe", ""),
    )}
    oral_hint = ("tablet" in product_tokens) or ("TAB" in unit_tokens)

    if explicit_iv and explicit_oral:
        return "ambiguous", f"explicit_iv={','.join(explicit_iv)}; explicit_oral={','.join(explicit_oral)}"
    if explicit_iv:
        return "iv", f"explicit_iv={','.join(explicit_iv)}"
    if explicit_oral:
        return "oral", f"explicit_oral={','.join(explicit_oral)}"
    if iv_hint and oral_hint:
        return "ambiguous", "conflicting_product_hints"
    if iv_hint:
        return "iv", "injectable_product_hint"
    if oral_hint:
        return "oral", "oral_product_hint"
    return "ambiguous", "missing_route_evidence"


def classify_admin_events(admins):
    admins = admins.copy()
    admins["event_txt"] = admins["event_txt"].astype("string").fillna("")
    admins["event_is_positive"] = admins["event_txt"].isin(POSITIVE_EVENT_TXT)

    route_labels = []
    route_reasons = []
    for row in admins.to_dict("records"):
        route_label, route_reason = classify_route(row)
        route_labels.append(route_label)
        route_reasons.append(route_reason)

    admins["route_label"] = route_labels
    admins["route_reason"] = route_reasons
    admins["treatment_event_time"] = admins["charttime"]
    admins = admins.sort_values(["subject_id", "hadm_id", "treatment_event_time", "emar_id"])
    return admins.reset_index(drop=True)


def build_pairs(studies, min_hours, max_hours):
    pair_rows = []

    studies = studies.sort_values(["subject_id", "study_time", "study_id", "cxr_id"], kind="stable")
    for subject_id, subject_studies in studies.groupby("subject_id", sort=False):
        subject_studies = subject_studies.reset_index(drop=True)
        for index in range(len(subject_studies) - 1):
            baseline = subject_studies.iloc[index]
            follow_up = subject_studies.iloc[index + 1]

            if baseline["hadm_id"] != follow_up["hadm_id"]:
                continue

            hours_diff = (follow_up["study_time"] - baseline["study_time"]).total_seconds() / 3600.0
            if not (min_hours <= hours_diff <= max_hours):
                continue

            pair_rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": baseline["hadm_id"],
                    "study_id_0": baseline["study_id"],
                    "study_id_1": follow_up["study_id"],
                    "cxr_0": baseline["cxr_id"],
                    "cxr_1": follow_up["cxr_id"],
                    "view_0": baseline["view_position"],
                    "view_1": follow_up["view_position"],
                    "t0": baseline["study_time"],
                    "t1": follow_up["study_time"],
                    "hours_diff": hours_diff,
                }
            )

    return pd.DataFrame(pair_rows)


def summarize_interval(events):
    summary = {}
    for label in ("iv", "oral", "ambiguous"):
        subset = events.loc[events["route_label"] == label]
        summary[f"{label}_count"] = int(len(subset))
        summary[f"{label}_first_time"] = (
            subset["treatment_event_time"].min().isoformat(sep=" ")
            if not subset.empty
            else ""
        )
        summary[f"{label}_last_time"] = (
            subset["treatment_event_time"].max().isoformat(sep=" ")
            if not subset.empty
            else ""
        )
        summary[f"{label}_event_types"] = "; ".join(sorted(subset["event_txt"].astype(str).unique()))
    return summary


def classify_pair_status(window_summary, washout_summary):
    between_iv = window_summary["iv_count"]
    between_oral = window_summary["oral_count"]
    between_ambiguous = window_summary["ambiguous_count"]

    washout_total = (
        washout_summary["iv_count"]
        + washout_summary["oral_count"]
        + washout_summary["ambiguous_count"]
    )

    if washout_total > 0:
        return "exclude_prior_furosemide"
    if between_iv > 0 and between_oral == 0 and between_ambiguous == 0:
        return "treated_iv_clean"
    if between_iv == 0 and between_oral == 0 and between_ambiguous == 0:
        return "control_clean"
    if between_iv > 0 and (between_oral > 0 or between_ambiguous > 0):
        return "exclude_mixed_route_furosemide"
    if between_iv == 0 and between_oral > 0 and between_ambiguous == 0:
        return "exclude_oral_only"
    if between_iv == 0 and between_oral == 0 and between_ambiguous > 0:
        return "exclude_ambiguous_only"
    return "exclude_other_furosemide"


def label_pairs(pairs, admins, washout_hours):
    positive_admins = admins[admins["event_is_positive"]].copy()
    grouped_admins = {
        (subject_id, hadm_id): frame.sort_values("treatment_event_time").reset_index(drop=True)
        for (subject_id, hadm_id), frame in positive_admins.groupby(["subject_id", "hadm_id"], sort=False)
    }

    labeled_rows = []
    washout_delta = pd.to_timedelta(washout_hours, unit="h")

    for row in pairs.itertuples(index=False):
        pair_dict = row._asdict()
        events = grouped_admins.get((pair_dict["subject_id"], pair_dict["hadm_id"]))

        if events is None:
            between_events = admins.iloc[0:0]
            washout_events = admins.iloc[0:0]
        else:
            between_events = events.loc[
                (events["treatment_event_time"] > pair_dict["t0"])
                & (events["treatment_event_time"] <= pair_dict["t1"])
            ]
            washout_events = events.loc[
                (events["treatment_event_time"] > pair_dict["t0"] - washout_delta)
                & (events["treatment_event_time"] <= pair_dict["t0"])
            ]

        between_summary = summarize_interval(between_events)
        washout_summary = summarize_interval(washout_events)
        label_status = classify_pair_status(between_summary, washout_summary)

        pair_dict.update(
            {
                "treated": label_status == "treated_iv_clean",
                "label_status": label_status,
                "between_iv_count": between_summary["iv_count"],
                "between_oral_count": between_summary["oral_count"],
                "between_ambiguous_count": between_summary["ambiguous_count"],
                "between_first_iv_time": between_summary["iv_first_time"],
                "between_last_iv_time": between_summary["iv_last_time"],
                "between_iv_event_types": between_summary["iv_event_types"],
                "washout_iv_count": washout_summary["iv_count"],
                "washout_oral_count": washout_summary["oral_count"],
                "washout_ambiguous_count": washout_summary["ambiguous_count"],
                "washout_first_iv_time": washout_summary["iv_first_time"],
                "washout_last_iv_time": washout_summary["iv_last_time"],
                "washout_iv_event_types": washout_summary["iv_event_types"],
            }
        )
        labeled_rows.append(pair_dict)

    if not labeled_rows:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "hadm_id",
                "study_id_0",
                "study_id_1",
                "cxr_0",
                "cxr_1",
                "view_0",
                "view_1",
                "t0",
                "t1",
                "hours_diff",
                "treated",
                "label_status",
            ]
        )

    labeled_pairs = pd.DataFrame(labeled_rows)
    labeled_pairs["treated"] = labeled_pairs["treated"].astype(bool)
    return labeled_pairs


def attach_image_availability(pairs, image_root):
    image_root = Path(image_root)
    pairs = pairs.copy()
    pairs["path_0"] = pairs["cxr_0"].map(lambda dicom_id: str(image_root / f"{dicom_id}.jpg"))
    pairs["path_1"] = pairs["cxr_1"].map(lambda dicom_id: str(image_root / f"{dicom_id}.jpg"))
    pairs["image_exists_0"] = pairs["path_0"].map(lambda path: Path(path).exists())
    pairs["image_exists_1"] = pairs["path_1"].map(lambda path: Path(path).exists())
    pairs["images_available"] = pairs["image_exists_0"] & pairs["image_exists_1"]
    return pairs


def write_outputs(args, admins, all_pairs, clean_pairs, summary):
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    admins_path = prefix.with_name(f"{prefix.name}_admin_events.csv")
    all_pairs_path = prefix.with_name(f"{prefix.name}_all.csv")
    clean_pairs_path = prefix.with_name(f"{prefix.name}_clean.csv")
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")

    admins.to_csv(admins_path, index=False)
    all_pairs.to_csv(all_pairs_path, index=False)
    clean_pairs.to_csv(clean_pairs_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Saved admin events to {admins_path}")
    print(f"Saved labeled pairs to {all_pairs_path}")
    print(f"Saved clean training pairs to {clean_pairs_path}")
    print(f"Saved summary to {summary_path}")


def build_summary(studies, admins, all_pairs, clean_pairs, cohort):
    route_counts = admins["route_label"].value_counts(dropna=False).to_dict()
    event_counts = admins["event_txt"].value_counts(dropna=False).to_dict()
    label_counts = (
        all_pairs["label_status"].value_counts(dropna=False).to_dict()
        if "label_status" in all_pairs.columns
        else {}
    )
    treated_clean_pairs = int(clean_pairs["treated"].sum()) if "treated" in clean_pairs.columns else 0
    control_clean_pairs = (
        int((~clean_pairs["treated"]).sum()) if "treated" in clean_pairs.columns else 0
    )

    return {
        "cohort": cohort,
        "studies": {
            "matched_frontal_studies": int(len(studies)),
            "subjects": int(studies["subject_id"].nunique()),
        },
        "administrations": {
            "furosemide_emar_rows": int(len(admins)),
            "positive_admin_events": int(admins["event_is_positive"].sum()),
            "route_label_counts": {str(key): int(value) for key, value in route_counts.items()},
            "event_txt_counts": {str(key): int(value) for key, value in event_counts.items()},
        },
        "pairs": {
            "all_candidate_pairs": int(len(all_pairs)),
            "clean_pairs": int(len(clean_pairs)),
            "treated_clean_pairs": treated_clean_pairs,
            "control_clean_pairs": control_clean_pairs,
            "label_status_counts": {str(key): int(value) for key, value in label_counts.items()},
        },
    }


def main():
    args = parse_args()

    admissions = load_admissions(args.admissions_path)
    studies = load_frontal_studies(args.metadata_path, admissions, args.admission_grace_hours)
    studies = filter_studies_for_cohort(studies, args)
    pairs = build_pairs(studies, args.min_hours, args.max_hours)

    admins = load_furosemide_medication_tables(args, admissions)
    admins = classify_admin_events(admins)

    all_pairs = label_pairs(pairs, admins, args.washout_hours)
    all_pairs = attach_image_availability(all_pairs, args.image_root)

    clean_pairs = all_pairs.loc[
        all_pairs["label_status"].isin(["treated_iv_clean", "control_clean"])
        & all_pairs["images_available"]
    ].copy()

    summary = build_summary(studies, admins, all_pairs, clean_pairs, args.cohort)
    write_outputs(args, admins, all_pairs, clean_pairs, summary)

    print(
        "[summary] all_pairs={all_pairs} clean_pairs={clean_pairs} treated={treated} control={control}".format(
            all_pairs=len(all_pairs),
            clean_pairs=len(clean_pairs),
            treated=int(clean_pairs["treated"].sum()) if "treated" in clean_pairs.columns else 0,
            control=int((~clean_pairs["treated"]).sum()) if "treated" in clean_pairs.columns else 0,
        )
    )


if __name__ == "__main__":
    main()
