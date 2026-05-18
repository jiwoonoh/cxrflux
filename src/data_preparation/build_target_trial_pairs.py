#!/usr/bin/env python3

import argparse
import gzip
import importlib.util
import io
import json
import re
import zlib
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDITED_BUILDER_PATH = Path(__file__).resolve().with_name("build_iv_furosemide_pairs.py")


def load_audited_builder():
    spec = importlib.util.spec_from_file_location(
        "audited_iv_furo_builder",
        AUDITED_BUILDER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load audited builder from {AUDITED_BUILDER_PATH}")
    spec.loader.exec_module(module)
    return module


audited = load_audited_builder()


LOOP_DRUG_PATTERNS = {
    "furosemide": re.compile(r"\b(?:furosemide|lasix)\b", re.IGNORECASE),
    "bumetanide": re.compile(r"\b(?:bumetanide|bumex)\b", re.IGNORECASE),
    "torsemide": re.compile(r"\b(?:torsemide|demadex)\b", re.IGNORECASE),
    "ethacrynic_acid": re.compile(r"\b(?:ethacrynic acid|edecrin)\b", re.IGNORECASE),
}
LOOP_DIURETIC_PATTERN = re.compile(
    "|".join(pattern.pattern for pattern in LOOP_DRUG_PATTERNS.values()),
    re.IGNORECASE,
)
IV_FUROSEMIDE_EQUIVALENT_FACTOR = {
    "furosemide": 1.0,
    "bumetanide": 20.0,
}
MG_UNIT_PATTERN = re.compile(r"\bmg\b", re.IGNORECASE)
MG_PER_ML_PATTERN = re.compile(
    r"(?P<mg>[0-9]+(?:\.[0-9]+)?)\s*mg\s*/\s*(?P<ml>[0-9]+(?:\.[0-9]+)?)\s*mL",
    re.IGNORECASE,
)
COMPETING_PROCEDURE_PATTERNS = {
    "dialysis": re.compile(r"dialysis|crrt|cvvh|cvvhd|cvvhdf|scuf|ultrafiltration", re.IGNORECASE),
    "chest_procedure": re.compile(r"thoracentesis|chest tube", re.IGNORECASE),
    "airway_intervention": re.compile(
        r"intubation|invasive ventilation|non-invasive ventilation|tracheostomy",
        re.IGNORECASE,
    ),
}


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a fixed-horizon target-trial cohort for HF IV furosemide analysis"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata-path", default="mimic-cxr-2.0.0-metadata.csv.gz")
    parser.add_argument(
        "--patients-path",
        default="physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz",
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
    parser.add_argument(
        "--icu-procedureevents-path",
        default="physionet.org/files/mimiciv/3.1/icu/procedureevents.csv.gz",
    )
    parser.add_argument(
        "--icu-d-items-path",
        default="physionet.org/files/mimiciv/3.1/icu/d_items.csv.gz",
    )
    parser.add_argument("--image-root", default="./mimic-cxr-jpg/")
    parser.add_argument("--chunksize", type=int, default=250000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-manifest", default=None)
    return parser.parse_args()


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required_sections = ["study_name", "cohort", "time_zero", "treatment", "follow_up"]
    missing = [key for key in required_sections if key not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    if config["cohort"].get("name") != "heart_failure":
        raise ValueError("This builder currently supports only the heart_failure cohort")
    if config["time_zero"].get("anchor") != "baseline_cxr":
        raise ValueError("This builder currently supports time_zero.anchor=baseline_cxr only")
    config["cohort"]["one_interval_per_admission"] = bool(
        config["cohort"].get("one_interval_per_admission", True)
    )

    treatment = config.setdefault("treatment", {})
    active_drugs = treatment.get("active_iv_loop_drugs")
    if active_drugs is None:
        active_drugs = ["furosemide"]
    active_drugs = [str(drug).strip().lower() for drug in active_drugs]
    unsupported = sorted(set(active_drugs) - set(LOOP_DRUG_PATTERNS))
    if unsupported:
        raise ValueError(f"Unsupported active IV loop drugs: {unsupported}")
    treatment["active_iv_loop_drugs"] = active_drugs
    treatment.setdefault(
        "treated_status_label",
        "treated_iv_furo_clean" if active_drugs == ["furosemide"] else "treated_iv_loop_clean",
    )

    return config


def build_output_paths(study_name, output_dir=None):
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = REPO_ROOT / "results" / study_name
    base_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base_dir": base_dir,
        "pairs_csv": base_dir / "target_trial_pairs_primary_h24.csv",
        "screening_csv": base_dir / "target_trial_pairs_primary_h24_screening.csv",
        "summary_json": base_dir / "target_trial_pairs_primary_h24_summary.json",
    }


def first_non_null_numeric(series):
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.dropna()
    return float(numeric.iloc[0]) if not numeric.empty else pd.NA


def attach_study_image_paths(studies, image_root):
    image_root = Path(image_root)
    studies = studies.copy()
    studies["path"] = studies["cxr_id"].map(lambda dicom_id: str(image_root / f"{dicom_id}.jpg"))
    studies["image_exists"] = studies["path"].map(lambda path: Path(path).exists())
    return studies


def load_patients(path):
    patients = pd.read_csv(
        path,
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        low_memory=False,
    )
    patients["subject_id"] = audited.normalize_id_series(patients["subject_id"])
    patients["anchor_age"] = pd.to_numeric(patients["anchor_age"], errors="coerce")
    patients["anchor_year"] = pd.to_numeric(patients["anchor_year"], errors="coerce")
    return patients.dropna(subset=["subject_id"]).reset_index(drop=True)


def compute_admission_age(admissions, patients):
    admissions = admissions.merge(patients, on="subject_id", how="left")
    admissions["admission_age"] = (
        admissions["anchor_age"]
        + admissions["admittime"].dt.year
        - admissions["anchor_year"]
    )
    return admissions


def aggregate_detail_rows(detail):
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "emar_id",
                "detail_routes",
                "detail_product_codes",
                "detail_product_descriptions",
                "detail_product_descriptions_other",
                "detail_product_units",
                "detail_administration_types",
                "detail_dose_given",
                "detail_dose_given_units",
                "detail_dose_due",
                "detail_dose_due_units",
                "detail_product_amount_given",
            ]
        )

    return (
        detail.groupby("emar_id", dropna=False)
        .agg(
            detail_routes=("route", audited.unique_join),
            detail_product_codes=("product_code", audited.unique_join),
            detail_product_descriptions=("product_description", audited.unique_join),
            detail_product_descriptions_other=("product_description_other", audited.unique_join),
            detail_product_units=("product_unit", audited.unique_join),
            detail_administration_types=("administration_type", audited.unique_join),
            detail_dose_given=("dose_given", first_non_null_numeric),
            detail_dose_given_units=("dose_given_unit", audited.unique_join),
            detail_dose_due=("dose_due", first_non_null_numeric),
            detail_dose_due_units=("dose_due_unit", audited.unique_join),
            detail_product_amount_given=("product_amount_given", first_non_null_numeric),
        )
        .reset_index()
    )


def aggregate_prescriptions(frame, key, suffix):
    if frame.empty:
        return pd.DataFrame(columns=[key])

    aggregated = (
        frame.groupby(key, dropna=False)
        .agg(
            **{
                f"prescription_routes{suffix}": ("route", audited.unique_join),
                f"prescription_drugs{suffix}": ("drug", audited.unique_join),
                f"prescription_codes{suffix}": ("formulary_drug_cd", audited.unique_join),
                f"prescription_strengths{suffix}": ("prod_strength", audited.unique_join),
                f"prescription_dose_val_rx{suffix}": ("dose_val_rx", first_non_null_numeric),
                f"prescription_dose_unit_rx{suffix}": ("dose_unit_rx", audited.unique_join),
                f"prescription_form_units{suffix}": ("form_unit_disp", audited.unique_join),
            }
        )
        .reset_index()
    )
    return aggregated


def load_loop_medication_tables(args, admissions):
    emar = audited.filter_in_chunks(
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
            LOOP_DIURETIC_PATTERN,
            na=False,
        ),
        chunksize=args.chunksize,
    )
    emar["subject_id"] = audited.normalize_id_series(emar["subject_id"])
    emar["hadm_id"] = audited.normalize_id_series(emar["hadm_id"])
    emar["emar_id"] = emar["emar_id"].astype("string").str.strip()
    emar["poe_id"] = emar["poe_id"].astype("string").str.strip()
    emar["pharmacy_id"] = (
        emar["pharmacy_id"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    emar["charttime"] = pd.to_datetime(emar["charttime"], errors="coerce")
    emar = emar.dropna(subset=["subject_id", "emar_id", "charttime"])

    missing_hadm_mask = emar["hadm_id"].isna() | (emar["hadm_id"] == "")
    if missing_hadm_mask.any():
        repaired = audited.assign_hadm_ids(
            emar.loc[missing_hadm_mask, ["subject_id", "emar_id", "charttime"]].rename(
                columns={"charttime": "event_time"}
            ),
            admissions,
            "event_time",
            6.0,
        )
        emar.loc[missing_hadm_mask, "hadm_id"] = repaired["hadm_id"].values

    emar = emar.dropna(subset=["hadm_id"]).copy()

    emar_ids = set(emar["emar_id"].dropna().astype(str))
    poe_ids = set(emar["poe_id"].dropna().astype(str)) - {"", "<NA>", "nan"}
    pharmacy_ids = set(emar["pharmacy_id"].dropna().astype(str)) - {"", "<NA>", "nan"}

    detail = audited.filter_in_chunks(
        args.emar_detail_path,
        usecols=[
            "emar_id",
            "route",
            "product_code",
            "product_description",
            "product_description_other",
            "product_unit",
            "administration_type",
            "dose_due",
            "dose_due_unit",
            "dose_given",
            "dose_given_unit",
            "product_amount_given",
        ],
        predicate=lambda chunk: chunk["emar_id"].astype("string").isin(emar_ids),
        chunksize=args.chunksize,
    )
    detail["emar_id"] = detail["emar_id"].astype("string").str.strip()
    detail_agg = aggregate_detail_rows(detail)

    pharmacy = audited.filter_in_chunks(
        args.pharmacy_path,
        usecols=["pharmacy_id", "poe_id", "route", "medication"],
        predicate=lambda chunk: chunk["medication"].astype("string").str.contains(
            LOOP_DIURETIC_PATTERN,
            na=False,
        )
        | chunk["pharmacy_id"].astype("string").isin(pharmacy_ids)
        | chunk["poe_id"].astype("string").isin(poe_ids),
        chunksize=args.chunksize,
    )
    pharmacy["pharmacy_id"] = (
        pharmacy["pharmacy_id"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    pharmacy["poe_id"] = pharmacy["poe_id"].astype("string").str.strip()
    pharmacy_agg = audited.aggregate_by_key(
        pharmacy,
        "pharmacy_id",
        {
            "route": "pharmacy_routes",
            "medication": "pharmacy_medications",
            "poe_id": "pharmacy_poe_ids",
        },
    )
    pharmacy_poe_agg = audited.aggregate_by_key(
        pharmacy,
        "poe_id",
        {
            "route": "pharmacy_routes_by_poe",
            "medication": "pharmacy_medications_by_poe",
        },
    )

    prescriptions = audited.filter_in_chunks(
        args.prescriptions_path,
        usecols=[
            "pharmacy_id",
            "poe_id",
            "route",
            "drug",
            "formulary_drug_cd",
            "prod_strength",
            "dose_val_rx",
            "dose_unit_rx",
            "form_unit_disp",
        ],
        predicate=lambda chunk: chunk["drug"].astype("string").str.contains(
            LOOP_DIURETIC_PATTERN,
            na=False,
        )
        | chunk["pharmacy_id"].astype("string").isin(pharmacy_ids)
        | chunk["poe_id"].astype("string").isin(poe_ids),
        chunksize=args.chunksize,
    )
    prescriptions["pharmacy_id"] = (
        prescriptions["pharmacy_id"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    prescriptions["poe_id"] = prescriptions["poe_id"].astype("string").str.strip()
    prescriptions_pharmacy_agg = aggregate_prescriptions(prescriptions, "pharmacy_id", "")
    prescriptions_poe_agg = aggregate_prescriptions(prescriptions, "poe_id", "_by_poe")

    poe_detail = audited.filter_in_chunks(
        args.poe_detail_path,
        usecols=["poe_id", "field_name", "field_value"],
        predicate=lambda chunk: chunk["poe_id"].astype("string").isin(poe_ids)
        & chunk["field_name"].astype("string").str.contains("route", case=False, na=False),
        chunksize=args.chunksize,
    )
    poe_detail["poe_id"] = poe_detail["poe_id"].astype("string").str.strip()
    poe_detail_agg = audited.aggregate_by_key(
        poe_detail,
        "poe_id",
        {"field_value": "poe_routes"},
    )

    admins = emar.merge(detail_agg, on="emar_id", how="left")
    admins = admins.merge(pharmacy_agg, on="pharmacy_id", how="left")
    admins = admins.merge(pharmacy_poe_agg, on="poe_id", how="left")
    admins = admins.merge(prescriptions_pharmacy_agg, on="pharmacy_id", how="left")
    admins = admins.merge(prescriptions_poe_agg, on="poe_id", how="left")
    admins = admins.merge(poe_detail_agg, on="poe_id", how="left")
    return admins


def combined_medication_text(row):
    return " | ".join(
        value
        for value in [
            row.get("medication", ""),
            row.get("pharmacy_medications", ""),
            row.get("pharmacy_medications_by_poe", ""),
            row.get("prescription_drugs", ""),
            row.get("prescription_drugs_by_poe", ""),
            row.get("detail_product_descriptions", ""),
            row.get("detail_product_descriptions_other", ""),
            row.get("prescription_strengths", ""),
            row.get("prescription_strengths_by_poe", ""),
        ]
        if audited.normalize_string(value)
    )


def detect_loop_drug(row):
    text = combined_medication_text(row)
    for label, pattern in LOOP_DRUG_PATTERNS.items():
        if pattern.search(text):
            return label
    return "unknown_loop"


def route_unit_contains_mg(value):
    return bool(MG_UNIT_PATTERN.search(audited.normalize_string(value)))


def infer_dose_mg(row):
    direct_candidates = [
        ("detail_dose_given", "detail_dose_given_units"),
        ("detail_dose_due", "detail_dose_due_units"),
        ("prescription_dose_val_rx", "prescription_dose_unit_rx"),
        ("prescription_dose_val_rx_by_poe", "prescription_dose_unit_rx_by_poe"),
    ]
    for value_col, unit_col in direct_candidates:
        dose_value = row.get(value_col, pd.NA)
        if pd.notna(dose_value) and route_unit_contains_mg(row.get(unit_col, "")):
            return float(dose_value), value_col

    amount = row.get("detail_product_amount_given", pd.NA)
    amount_unit = row.get("detail_product_units", "")
    if pd.notna(amount) and "ML" in audited.normalize_string(amount_unit).upper():
        product_text = combined_medication_text(row)
        match = MG_PER_ML_PATTERN.search(product_text)
        if match:
            mg = float(match.group("mg"))
            ml = float(match.group("ml"))
            if ml > 0:
                return float(amount) * (mg / ml), "detail_product_amount_given_x_product_strength"

    return pd.NA, ""


def classify_loop_admin_events(admins):
    admins = admins.copy()
    admins["event_txt"] = admins["event_txt"].astype("string").fillna("")
    admins["event_is_positive"] = admins["event_txt"].isin(audited.POSITIVE_EVENT_TXT)

    route_labels = []
    route_reasons = []
    loop_drugs = []
    inferred_doses = []
    inferred_sources = []

    for row in admins.to_dict("records"):
        route_label, route_reason = audited.classify_route(row)
        loop_drug = detect_loop_drug(row)
        inferred_dose_mg, inferred_dose_source = infer_dose_mg(row)
        route_labels.append(route_label)
        route_reasons.append(route_reason)
        loop_drugs.append(loop_drug)
        inferred_doses.append(inferred_dose_mg)
        inferred_sources.append(inferred_dose_source)

    admins["route_label"] = route_labels
    admins["route_reason"] = route_reasons
    admins["loop_drug_label"] = loop_drugs
    admins["inferred_dose_mg"] = inferred_doses
    admins["inferred_dose_source"] = inferred_sources
    admins["treatment_event_time"] = admins["charttime"]
    admins = admins.sort_values(
        ["subject_id", "hadm_id", "treatment_event_time", "emar_id"],
        kind="stable",
    )
    return admins.reset_index(drop=True)


def summarize_loop_window(events, active_iv_loop_drugs=None):
    active_iv_loop_drugs = set(active_iv_loop_drugs or ["furosemide"])
    summary = {
        "iv_any_count": 0,
        "iv_furo_count": 0,
        "iv_bumetanide_count": 0,
        "iv_other_loop_count": 0,
        "iv_active_loop_count": 0,
        "iv_inactive_loop_count": 0,
        "oral_loop_count": 0,
        "ambiguous_loop_count": 0,
        "first_any_loop_time": "",
        "first_iv_furo_time": "",
        "first_iv_bumetanide_time": "",
        "first_iv_active_loop_time": "",
        "first_iv_other_loop_time": "",
        "iv_other_loop_drugs": "",
        "iv_active_loop_drugs": "",
        "iv_inactive_loop_drugs": "",
        "furo_admin_count": 0,
        "furo_dose_total_mg": 0.0,
        "furo_dose_event_count_with_mg": 0,
        "furo_dose_event_count_missing_mg": 0,
        "bumetanide_admin_count": 0,
        "bumetanide_dose_total_mg": 0.0,
        "bumetanide_furosemide_equiv_total_mg": 0.0,
        "bumetanide_dose_event_count_with_mg": 0,
        "bumetanide_dose_event_count_missing_mg": 0,
        "active_loop_furosemide_equiv_total_mg": 0.0,
        "active_loop_equiv_event_count_with_mg": 0,
        "active_loop_equiv_event_count_missing_mg": 0,
    }
    if events.empty:
        return summary

    events = events.sort_values("treatment_event_time", kind="stable")
    iv_events = events.loc[events["route_label"] == "iv"]
    oral_events = events.loc[events["route_label"] == "oral"]
    ambiguous_events = events.loc[events["route_label"] == "ambiguous"]
    iv_furo = iv_events.loc[iv_events["loop_drug_label"] == "furosemide"]
    iv_bumetanide = iv_events.loc[iv_events["loop_drug_label"] == "bumetanide"]
    iv_active = iv_events.loc[iv_events["loop_drug_label"].isin(active_iv_loop_drugs)]
    iv_inactive = iv_events.loc[~iv_events["loop_drug_label"].isin(active_iv_loop_drugs)]
    iv_other = iv_events.loc[iv_events["loop_drug_label"] != "furosemide"]

    summary["iv_any_count"] = int(len(iv_events))
    summary["iv_furo_count"] = int(len(iv_furo))
    summary["iv_bumetanide_count"] = int(len(iv_bumetanide))
    summary["iv_other_loop_count"] = int(len(iv_other))
    summary["iv_active_loop_count"] = int(len(iv_active))
    summary["iv_inactive_loop_count"] = int(len(iv_inactive))
    summary["oral_loop_count"] = int(len(oral_events))
    summary["ambiguous_loop_count"] = int(len(ambiguous_events))
    summary["first_any_loop_time"] = events["treatment_event_time"].min().isoformat(sep=" ")
    if not iv_furo.empty:
        summary["first_iv_furo_time"] = iv_furo["treatment_event_time"].min().isoformat(sep=" ")
        furo_doses = pd.to_numeric(iv_furo["inferred_dose_mg"], errors="coerce")
        summary["furo_admin_count"] = int(len(iv_furo))
        summary["furo_dose_total_mg"] = float(furo_doses.dropna().sum()) if furo_doses.notna().any() else 0.0
        summary["furo_dose_event_count_with_mg"] = int(furo_doses.notna().sum())
        summary["furo_dose_event_count_missing_mg"] = int(furo_doses.isna().sum())
    if not iv_bumetanide.empty:
        summary["first_iv_bumetanide_time"] = iv_bumetanide["treatment_event_time"].min().isoformat(sep=" ")
        bumetanide_doses = pd.to_numeric(iv_bumetanide["inferred_dose_mg"], errors="coerce")
        summary["bumetanide_admin_count"] = int(len(iv_bumetanide))
        summary["bumetanide_dose_total_mg"] = (
            float(bumetanide_doses.dropna().sum()) if bumetanide_doses.notna().any() else 0.0
        )
        summary["bumetanide_furosemide_equiv_total_mg"] = (
            float(bumetanide_doses.dropna().sum()) * IV_FUROSEMIDE_EQUIVALENT_FACTOR["bumetanide"]
            if bumetanide_doses.notna().any()
            else 0.0
        )
        summary["bumetanide_dose_event_count_with_mg"] = int(bumetanide_doses.notna().sum())
        summary["bumetanide_dose_event_count_missing_mg"] = int(bumetanide_doses.isna().sum())
    if not iv_active.empty:
        summary["first_iv_active_loop_time"] = iv_active["treatment_event_time"].min().isoformat(sep=" ")
        summary["iv_active_loop_drugs"] = "; ".join(sorted(iv_active["loop_drug_label"].dropna().unique()))
        equiv_values = []
        missing_equiv_count = 0
        for row in iv_active.to_dict("records"):
            dose = pd.to_numeric(pd.Series([row.get("inferred_dose_mg")]), errors="coerce").iloc[0]
            factor = IV_FUROSEMIDE_EQUIVALENT_FACTOR.get(row.get("loop_drug_label"))
            if pd.isna(dose) or factor is None:
                missing_equiv_count += 1
            else:
                equiv_values.append(float(dose) * factor)
        summary["active_loop_furosemide_equiv_total_mg"] = float(sum(equiv_values))
        summary["active_loop_equiv_event_count_with_mg"] = int(len(equiv_values))
        summary["active_loop_equiv_event_count_missing_mg"] = int(missing_equiv_count)
    if not iv_inactive.empty:
        summary["iv_inactive_loop_drugs"] = "; ".join(sorted(iv_inactive["loop_drug_label"].dropna().unique()))
    if not iv_other.empty:
        summary["first_iv_other_loop_time"] = iv_other["treatment_event_time"].min().isoformat(sep=" ")
        summary["iv_other_loop_drugs"] = "; ".join(sorted(iv_other["loop_drug_label"].dropna().unique()))
    return summary


def classify_treatment_status(grace_summary, washout_summary, active_status_label="treated_iv_furo_clean"):
    if washout_summary["iv_any_count"] > 0:
        return "exclude_prior_iv_loop_washout"
    if washout_summary["oral_loop_count"] > 0:
        return "exclude_prior_oral_loop_washout"
    if washout_summary["ambiguous_loop_count"] > 0:
        return "exclude_prior_ambiguous_loop_washout"

    if (
        grace_summary["iv_active_loop_count"] > 0
        and grace_summary["iv_inactive_loop_count"] == 0
        and grace_summary["oral_loop_count"] == 0
        and grace_summary["ambiguous_loop_count"] == 0
    ):
        return active_status_label
    if (
        grace_summary["iv_active_loop_count"] == 0
        and grace_summary["iv_inactive_loop_count"] == 0
        and grace_summary["oral_loop_count"] == 0
        and grace_summary["ambiguous_loop_count"] == 0
    ):
        return "control_clean"
    if (
        grace_summary["iv_active_loop_count"] == 0
        and grace_summary["iv_inactive_loop_count"] > 0
        and grace_summary["oral_loop_count"] == 0
        and grace_summary["ambiguous_loop_count"] == 0
    ):
        return "exclude_other_iv_loop_in_grace"
    if grace_summary["iv_active_loop_count"] > 0 and (
        grace_summary["iv_inactive_loop_count"] > 0
        or grace_summary["oral_loop_count"] > 0
        or grace_summary["ambiguous_loop_count"] > 0
    ):
        return "exclude_mixed_loop_in_grace"
    if (
        grace_summary["iv_active_loop_count"] == 0
        and grace_summary["iv_inactive_loop_count"] == 0
        and grace_summary["oral_loop_count"] > 0
        and grace_summary["ambiguous_loop_count"] == 0
    ):
        return "exclude_oral_loop_in_grace"
    if (
        grace_summary["iv_active_loop_count"] == 0
        and grace_summary["iv_inactive_loop_count"] == 0
        and grace_summary["oral_loop_count"] == 0
        and grace_summary["ambiguous_loop_count"] > 0
    ):
        return "exclude_ambiguous_loop_in_grace"
    return "exclude_other_loop_in_grace"


def load_competing_procedure_events(args):
    d_items = read_csv_resilient(
        args.icu_d_items_path,
        usecols=["itemid", "label", "linksto"],
        low_memory=False,
    )
    d_items = d_items[d_items["linksto"] == "procedureevents"].copy()
    item_groups = {}
    for row in d_items.itertuples(index=False):
        label = audited.normalize_string(row.label)
        matches = [
            group
            for group, pattern in COMPETING_PROCEDURE_PATTERNS.items()
            if pattern.search(label)
        ]
        if matches:
            item_groups[int(row.itemid)] = {
                "label": label,
                "groups": matches,
            }

    if not item_groups:
        return pd.DataFrame(
            columns=["subject_id", "hadm_id", "event_time", "procedure_label", "competing_group"]
        )

    procedureevents = read_csv_resilient(
        args.icu_procedureevents_path,
        usecols=["subject_id", "hadm_id", "itemid", "starttime", "endtime"],
        low_memory=False,
    )
    procedureevents = procedureevents.loc[procedureevents["itemid"].isin(item_groups)].copy()
    if procedureevents.empty:
        return pd.DataFrame(
            columns=["subject_id", "hadm_id", "event_time", "procedure_label", "competing_group"]
        )

    procedureevents["subject_id"] = audited.normalize_id_series(procedureevents["subject_id"])
    procedureevents["hadm_id"] = audited.normalize_id_series(procedureevents["hadm_id"])
    procedureevents["starttime"] = pd.to_datetime(procedureevents["starttime"], errors="coerce")
    procedureevents["endtime"] = pd.to_datetime(procedureevents["endtime"], errors="coerce")
    procedureevents["event_time"] = procedureevents["starttime"].fillna(procedureevents["endtime"])
    procedureevents = procedureevents.dropna(subset=["subject_id", "hadm_id", "event_time"])

    exploded_rows = []
    for row in procedureevents.itertuples(index=False):
        item_metadata = item_groups.get(int(row.itemid))
        if item_metadata is None:
            continue
        for group in item_metadata["groups"]:
            exploded_rows.append(
                {
                    "subject_id": row.subject_id,
                    "hadm_id": row.hadm_id,
                    "event_time": row.event_time,
                    "procedure_label": item_metadata["label"],
                    "competing_group": group,
                }
            )

    return pd.DataFrame(exploded_rows)


def summarize_competing_window(events):
    summary = {
        "dialysis_count": 0,
        "chest_procedure_count": 0,
        "airway_intervention_count": 0,
        "any_competing_count": 0,
        "any_competing": False,
        "first_competing_time": "",
        "competing_groups": "",
        "competing_labels": "",
    }
    if events.empty:
        return summary

    summary["dialysis_count"] = int((events["competing_group"] == "dialysis").sum())
    summary["chest_procedure_count"] = int((events["competing_group"] == "chest_procedure").sum())
    summary["airway_intervention_count"] = int((events["competing_group"] == "airway_intervention").sum())
    summary["any_competing_count"] = int(len(events))
    summary["any_competing"] = True
    summary["first_competing_time"] = events["event_time"].min().isoformat(sep=" ")
    summary["competing_groups"] = "; ".join(sorted(events["competing_group"].unique()))
    summary["competing_labels"] = "; ".join(sorted(events["procedure_label"].unique()))
    return summary


def select_follow_up(studies, baseline_time, target_time, window_start, window_end, require_image):
    candidates = studies.loc[
        (studies["study_time"] >= window_start)
        & (studies["study_time"] <= window_end)
        & (studies["study_time"] > baseline_time)
    ].copy()
    if require_image:
        candidates = candidates.loc[candidates["image_exists"]].copy()
    if candidates.empty:
        return None

    candidates["abs_error_hours"] = (
        (candidates["study_time"] - target_time).abs().dt.total_seconds() / 3600.0
    )
    candidates = candidates.sort_values(
        ["abs_error_hours", "study_time", "study_id", "cxr_id"],
        kind="stable",
    )
    return candidates.iloc[0]


def compute_screening_for_candidate(
    study_row,
    study_rank,
    treatment_events,
    competing_events,
    config,
):
    grace_start = pd.to_timedelta(config["treatment"]["grace_window_hours"][0], unit="h")
    grace_end = pd.to_timedelta(config["treatment"]["grace_window_hours"][1], unit="h")
    washout_delta = pd.to_timedelta(config["treatment"]["washout_hours"], unit="h")
    follow_up_window = config["follow_up"]["allowable_window_hours"]
    target_hour = config["follow_up"]["target_hour"]
    active_iv_loop_drugs = config["treatment"]["active_iv_loop_drugs"]
    treated_status_label = config["treatment"]["treated_status_label"]

    t0 = study_row["study_time"]
    grace_window_start = t0 + grace_start
    grace_window_end = t0 + grace_end
    horizon_target_time = t0 + pd.to_timedelta(target_hour, unit="h")
    horizon_window_start = t0 + pd.to_timedelta(follow_up_window[0], unit="h")
    horizon_window_end = t0 + pd.to_timedelta(follow_up_window[1], unit="h")

    washout_events = treatment_events.loc[
        (treatment_events["treatment_event_time"] > t0 - washout_delta)
        & (treatment_events["treatment_event_time"] <= t0)
    ]
    grace_events = treatment_events.loc[
        (treatment_events["treatment_event_time"] >= grace_window_start)
        & (treatment_events["treatment_event_time"] <= grace_window_end)
    ]
    post_grace_events = treatment_events.loc[
        (treatment_events["treatment_event_time"] > grace_window_end)
        & (treatment_events["treatment_event_time"] <= horizon_window_end)
    ]
    competing_window_events = competing_events.loc[
        (competing_events["event_time"] > t0)
        & (competing_events["event_time"] <= horizon_window_end)
    ]

    washout_summary = summarize_loop_window(washout_events, active_iv_loop_drugs)
    grace_summary = summarize_loop_window(grace_events, active_iv_loop_drugs)
    post_grace_summary = summarize_loop_window(post_grace_events, active_iv_loop_drugs)
    competing_summary = summarize_competing_window(competing_window_events)
    treatment_status = classify_treatment_status(grace_summary, washout_summary, treated_status_label)

    screening_reason = ""
    if not bool(study_row["image_exists"]):
        screening_reason = "baseline_image_missing"
    elif study_row["admission_age"] < 18:
        screening_reason = "age_under_18"
    elif treatment_status not in {treated_status_label, "control_clean"}:
        screening_reason = treatment_status
    elif competing_summary["any_competing"]:
        screening_reason = "exclude_competing_intervention_before_horizon"

    return {
        "subject_id": study_row["subject_id"],
        "hadm_id": study_row["hadm_id"],
        "candidate_rank_in_admission": study_rank,
        "study_id_0": study_row["study_id"],
        "cxr_0": study_row["cxr_id"],
        "view_0": study_row["view_position"],
        "t0": t0,
        "path_0": study_row["path"],
        "image_exists_0": bool(study_row["image_exists"]),
        "admittime": study_row["admittime"],
        "dischtime": study_row["dischtime"],
        "gender": study_row["gender"],
        "age_at_t0": float(study_row["admission_age"]) if pd.notna(study_row["admission_age"]) else pd.NA,
        "hours_since_admission_at_t0": (
            (t0 - study_row["admittime"]).total_seconds() / 3600.0
            if pd.notna(study_row["admittime"])
            else pd.NA
        ),
        "grace_window_start": grace_window_start,
        "grace_window_end": grace_window_end,
        "target_follow_up_time": horizon_target_time,
        "follow_up_window_start": horizon_window_start,
        "follow_up_window_end": horizon_window_end,
        "washout_iv_any_count": washout_summary["iv_any_count"],
        "washout_oral_loop_count": washout_summary["oral_loop_count"],
        "washout_ambiguous_loop_count": washout_summary["ambiguous_loop_count"],
        "grace_iv_furo_count": grace_summary["iv_furo_count"],
        "grace_iv_bumetanide_count": grace_summary["iv_bumetanide_count"],
        "grace_iv_other_loop_count": grace_summary["iv_other_loop_count"],
        "grace_iv_active_loop_count": grace_summary["iv_active_loop_count"],
        "grace_iv_inactive_loop_count": grace_summary["iv_inactive_loop_count"],
        "grace_oral_loop_count": grace_summary["oral_loop_count"],
        "grace_ambiguous_loop_count": grace_summary["ambiguous_loop_count"],
        "grace_first_iv_furo_time": grace_summary["first_iv_furo_time"],
        "grace_first_iv_bumetanide_time": grace_summary["first_iv_bumetanide_time"],
        "grace_first_iv_active_loop_time": grace_summary["first_iv_active_loop_time"],
        "grace_first_iv_other_loop_time": grace_summary["first_iv_other_loop_time"],
        "grace_iv_other_loop_drugs": grace_summary["iv_other_loop_drugs"],
        "grace_iv_active_loop_drugs": grace_summary["iv_active_loop_drugs"],
        "grace_iv_inactive_loop_drugs": grace_summary["iv_inactive_loop_drugs"],
        "grace_furo_admin_count": grace_summary["furo_admin_count"],
        "grace_furo_dose_total_mg": grace_summary["furo_dose_total_mg"],
        "grace_furo_dose_event_count_with_mg": grace_summary["furo_dose_event_count_with_mg"],
        "grace_furo_dose_event_count_missing_mg": grace_summary["furo_dose_event_count_missing_mg"],
        "grace_bumetanide_admin_count": grace_summary["bumetanide_admin_count"],
        "grace_bumetanide_dose_total_mg": grace_summary["bumetanide_dose_total_mg"],
        "grace_bumetanide_furosemide_equiv_total_mg": grace_summary["bumetanide_furosemide_equiv_total_mg"],
        "grace_active_loop_furosemide_equiv_total_mg": grace_summary["active_loop_furosemide_equiv_total_mg"],
        "grace_active_loop_equiv_event_count_with_mg": grace_summary["active_loop_equiv_event_count_with_mg"],
        "grace_active_loop_equiv_event_count_missing_mg": grace_summary["active_loop_equiv_event_count_missing_mg"],
        "post_grace_iv_furo_count": post_grace_summary["iv_furo_count"],
        "post_grace_iv_bumetanide_count": post_grace_summary["iv_bumetanide_count"],
        "post_grace_iv_other_loop_count": post_grace_summary["iv_other_loop_count"],
        "post_grace_iv_active_loop_count": post_grace_summary["iv_active_loop_count"],
        "post_grace_iv_inactive_loop_count": post_grace_summary["iv_inactive_loop_count"],
        "post_grace_oral_loop_count": post_grace_summary["oral_loop_count"],
        "post_grace_ambiguous_loop_count": post_grace_summary["ambiguous_loop_count"],
        "post_grace_iv_other_loop_drugs": post_grace_summary["iv_other_loop_drugs"],
        "post_grace_iv_active_loop_drugs": post_grace_summary["iv_active_loop_drugs"],
        "post_grace_iv_inactive_loop_drugs": post_grace_summary["iv_inactive_loop_drugs"],
        "post_grace_furo_admin_count": post_grace_summary["furo_admin_count"],
        "post_grace_furo_dose_total_mg": post_grace_summary["furo_dose_total_mg"],
        "post_grace_bumetanide_admin_count": post_grace_summary["bumetanide_admin_count"],
        "post_grace_bumetanide_dose_total_mg": post_grace_summary["bumetanide_dose_total_mg"],
        "post_grace_bumetanide_furosemide_equiv_total_mg": post_grace_summary["bumetanide_furosemide_equiv_total_mg"],
        "post_grace_active_loop_furosemide_equiv_total_mg": post_grace_summary["active_loop_furosemide_equiv_total_mg"],
        "strategy_deviation_before_horizon": (
            post_grace_summary["iv_any_count"]
            + post_grace_summary["oral_loop_count"]
            + post_grace_summary["ambiguous_loop_count"]
        )
        > 0,
        "treatment_status": treatment_status,
        "treated": treatment_status == treated_status_label,
        "competing_dialysis_count": competing_summary["dialysis_count"],
        "competing_chest_procedure_count": competing_summary["chest_procedure_count"],
        "competing_airway_intervention_count": competing_summary["airway_intervention_count"],
        "competing_any_before_horizon": competing_summary["any_competing"],
        "first_competing_time": competing_summary["first_competing_time"],
        "competing_groups": competing_summary["competing_groups"],
        "competing_labels": competing_summary["competing_labels"],
        "screening_reason": screening_reason or "eligible",
        "candidate_selected": screening_reason == "",
    }


def build_target_trial_pairs(studies, admins, competing_events, config):
    one_interval_per_admission = bool(config["cohort"].get("one_interval_per_admission", True))
    positive_admins = admins.loc[admins["event_is_positive"]].copy()
    treatment_groups = {
        key: frame.sort_values("treatment_event_time", kind="stable").reset_index(drop=True)
        for key, frame in positive_admins.groupby(["subject_id", "hadm_id"], sort=False)
    }
    competing_groups = {
        key: frame.sort_values("event_time", kind="stable").reset_index(drop=True)
        for key, frame in competing_events.groupby(["subject_id", "hadm_id"], sort=False)
    }

    screening_rows = []
    selected_rows = []
    screening_reason_counts = Counter()

    grouped_studies = studies.sort_values(
        ["subject_id", "hadm_id", "study_time", "study_id", "cxr_id"],
        kind="stable",
    ).groupby(["subject_id", "hadm_id"], sort=False)

    for (subject_id, hadm_id), admission_studies in grouped_studies:
        admission_studies = admission_studies.reset_index(drop=True)
        treatment_events = treatment_groups.get(
            (subject_id, hadm_id),
            positive_admins.iloc[0:0],
        )
        competing_window = competing_groups.get(
            (subject_id, hadm_id),
            competing_events.iloc[0:0],
        )

        selected_candidate = None
        for candidate_rank, (_, study_row) in enumerate(admission_studies.iterrows(), start=1):
            screening_row = compute_screening_for_candidate(
                study_row,
                candidate_rank,
                treatment_events,
                competing_window,
                config,
            )
            screening_rows.append(screening_row)
            screening_reason_counts[screening_row["screening_reason"]] += 1
            if screening_row["candidate_selected"]:
                selected_candidate = screening_row
                target_time = selected_candidate["target_follow_up_time"]
                window_start = selected_candidate["follow_up_window_start"]
                window_end = selected_candidate["follow_up_window_end"]

                follow_up_any = select_follow_up(
                    admission_studies,
                    selected_candidate["t0"],
                    target_time,
                    window_start,
                    window_end,
                    require_image=False,
                )
                follow_up_image = select_follow_up(
                    admission_studies,
                    selected_candidate["t0"],
                    target_time,
                    window_start,
                    window_end,
                    require_image=True,
                )

                row = dict(selected_candidate)
                row["primary_analysis_eligible"] = True
                row["discharged_before_horizon_end"] = (
                    pd.notna(selected_candidate["dischtime"])
                    and selected_candidate["dischtime"] < selected_candidate["follow_up_window_end"]
                )
                row["follow_up_study_observed"] = follow_up_any is not None
                row["follow_up_image_observed"] = follow_up_image is not None

                if follow_up_any is not None:
                    row["follow_up_any_study_id"] = follow_up_any["study_id"]
                    row["follow_up_any_cxr_id"] = follow_up_any["cxr_id"]
                    row["follow_up_any_time"] = follow_up_any["study_time"]
                    row["follow_up_any_hours"] = (
                        (follow_up_any["study_time"] - selected_candidate["t0"]).total_seconds() / 3600.0
                    )
                else:
                    row["follow_up_any_study_id"] = pd.NA
                    row["follow_up_any_cxr_id"] = pd.NA
                    row["follow_up_any_time"] = pd.NaT
                    row["follow_up_any_hours"] = pd.NA

                if follow_up_image is not None:
                    row["study_id_1"] = follow_up_image["study_id"]
                    row["cxr_1"] = follow_up_image["cxr_id"]
                    row["view_1"] = follow_up_image["view_position"]
                    row["t1"] = follow_up_image["study_time"]
                    row["hours_diff"] = (
                        (follow_up_image["study_time"] - selected_candidate["t0"]).total_seconds() / 3600.0
                    )
                    row["path_1"] = follow_up_image["path"]
                    row["image_exists_1"] = bool(follow_up_image["image_exists"])
                    row["follow_up_selection_status"] = "image_observed"
                elif follow_up_any is not None:
                    row["study_id_1"] = pd.NA
                    row["cxr_1"] = pd.NA
                    row["view_1"] = pd.NA
                    row["t1"] = pd.NaT
                    row["hours_diff"] = pd.NA
                    row["path_1"] = pd.NA
                    row["image_exists_1"] = False
                    row["follow_up_selection_status"] = "study_in_window_missing_local_image"
                else:
                    row["study_id_1"] = pd.NA
                    row["cxr_1"] = pd.NA
                    row["view_1"] = pd.NA
                    row["t1"] = pd.NaT
                    row["hours_diff"] = pd.NA
                    row["path_1"] = pd.NA
                    row["image_exists_1"] = False
                    row["follow_up_selection_status"] = "no_study_in_window"

                selected_rows.append(row)
                if one_interval_per_admission:
                    break

    screening_df = pd.DataFrame(screening_rows)
    selected_df = pd.DataFrame(selected_rows)
    return selected_df, screening_df, screening_reason_counts


def build_summary(config, studies, admins, competing_events, pairs, screening, screening_reason_counts):
    treated_pairs = int(pairs["treated"].sum()) if not pairs.empty else 0
    control_pairs = int((~pairs["treated"]).sum()) if not pairs.empty else 0

    follow_up_image_counts = (
        pairs.groupby("treated")["follow_up_image_observed"].sum().to_dict()
        if not pairs.empty
        else {}
    )
    deviation_counts = (
        pairs.assign(
            any_post_grace_loop=lambda frame: (
                frame["post_grace_iv_furo_count"]
                + frame["post_grace_iv_other_loop_count"]
                + frame["post_grace_oral_loop_count"]
                + frame["post_grace_ambiguous_loop_count"]
            )
            > 0
        )
        .groupby("treated")["any_post_grace_loop"]
        .sum()
        .to_dict()
        if not pairs.empty
        else {}
    )

    return {
        "study_name": config["study_name"],
        "config": config,
        "source_notes": {
            "treatment_labels_reuse_audited_route_logic": True,
            "competing_interventions_source": "icu.procedureevents exact timestamps",
            "follow_up_observation_requires_local_image_for_image_outcome": True,
        },
        "counts": {
            "matched_frontal_studies": int(len(studies)),
            "hf_admissions_with_candidate_studies": int(studies["hadm_id"].nunique()) if not studies.empty else 0,
            "loop_admin_rows": int(len(admins)),
            "positive_loop_admin_rows": int(admins["event_is_positive"].sum()) if not admins.empty else 0,
            "competing_procedure_rows": int(len(competing_events)),
            "screened_candidate_baselines": int(len(screening)),
            "selected_primary_baselines": int(len(pairs)),
            "treated_selected_primary_baselines": treated_pairs,
            "control_selected_primary_baselines": control_pairs,
            "follow_up_image_observed": int(pairs["follow_up_image_observed"].sum()) if not pairs.empty else 0,
            "follow_up_study_observed": int(pairs["follow_up_study_observed"].sum()) if not pairs.empty else 0,
        },
        "screening_reason_counts": {
            key: int(value) for key, value in sorted(screening_reason_counts.items())
        },
        "selected_follow_up_image_by_treatment": {
            str(key): int(value) for key, value in follow_up_image_counts.items()
        },
        "selected_post_grace_strategy_deviation_by_treatment": {
            str(key): int(value) for key, value in deviation_counts.items()
        },
    }


def write_outputs(output_paths, pairs, screening, summary, output_manifest=None):
    pairs.to_csv(output_paths["pairs_csv"], index=False)
    screening.to_csv(output_paths["screening_csv"], index=False)
    with output_paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    manifest = {
        "pairs_csv": str(output_paths["pairs_csv"]),
        "screening_csv": str(output_paths["screening_csv"]),
        "summary_json": str(output_paths["summary_json"]),
        "selected_primary_baselines": summary["counts"]["selected_primary_baselines"],
        "treated_selected_primary_baselines": summary["counts"]["treated_selected_primary_baselines"],
        "control_selected_primary_baselines": summary["counts"]["control_selected_primary_baselines"],
    }
    if output_manifest:
        output_manifest_path = Path(output_manifest)
        output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        output_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    output_paths = build_output_paths(config["study_name"], args.output_dir)

    print("[stage] loading admissions and patients")
    admissions = audited.load_admissions(args.admissions_path)
    patients = load_patients(args.patients_path)
    admissions = compute_admission_age(admissions, patients)

    print("[stage] loading frontal studies and heart-failure admissions")
    studies = audited.load_frontal_studies(
        args.metadata_path,
        admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
        6.0,
    )
    heart_failure_hadm_ids = audited.load_heart_failure_hadm_ids(
        args.diagnoses_path,
        args.diagnosis_titles_path,
        args.chunksize,
    )
    studies = studies.loc[studies["hadm_id"].isin(heart_failure_hadm_ids)].copy()
    studies = studies.merge(
        admissions[["subject_id", "hadm_id", "admittime", "dischtime", "gender", "admission_age"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )
    studies = studies.loc[studies["admission_age"] >= 18].copy()
    studies = attach_study_image_paths(studies, args.image_root)

    print("[stage] loading loop diuretic administrations")
    admins = load_loop_medication_tables(args, admissions)
    admins = classify_loop_admin_events(admins)
    admins = admins.loc[admins["hadm_id"].isin(heart_failure_hadm_ids)].copy()

    print("[stage] loading competing ICU procedure events")
    competing_events = load_competing_procedure_events(args)
    if not competing_events.empty:
        competing_events = competing_events.loc[
            competing_events["hadm_id"].isin(heart_failure_hadm_ids)
        ].copy()

    print("[stage] selecting target-trial baselines")
    pairs, screening, screening_reason_counts = build_target_trial_pairs(
        studies,
        admins,
        competing_events,
        config,
    )
    print("[stage] writing outputs")
    summary = build_summary(
        config,
        studies,
        admins,
        competing_events,
        pairs,
        screening,
        screening_reason_counts,
    )
    write_outputs(output_paths, pairs, screening, summary, args.output_manifest)

    print(f"Saved target-trial pairs to {output_paths['pairs_csv']}")
    print(f"Saved screening audit to {output_paths['screening_csv']}")
    print(f"Saved summary to {output_paths['summary_json']}")
    print(
        "[summary] selected={selected} treated={treated} control={control} follow_up_image={follow_up_image}".format(
            selected=summary["counts"]["selected_primary_baselines"],
            treated=summary["counts"]["treated_selected_primary_baselines"],
            control=summary["counts"]["control_selected_primary_baselines"],
            follow_up_image=summary["counts"]["follow_up_image_observed"],
        )
    )


if __name__ == "__main__":
    main()
