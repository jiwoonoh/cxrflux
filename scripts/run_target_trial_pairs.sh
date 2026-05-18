#!/bin/bash
#SBATCH --job-name=tt_pairs
#SBATCH --account=ntrayan1_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=tt_pairs_%j.out

set -euo pipefail

FINAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FINAL_DIR}"

CONFIG_PATH="${CONFIG_PATH:-${FINAL_DIR}/configs/primary_h24.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-}"

METADATA_PATH="${METADATA_PATH:-mimic-cxr-2.0.0-metadata.csv.gz}"
PATIENTS_PATH="${PATIENTS_PATH:-physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz}"
ADMISSIONS_PATH="${ADMISSIONS_PATH:-physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz}"
DIAGNOSES_PATH="${DIAGNOSES_PATH:-physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz}"
DIAGNOSIS_TITLES_PATH="${DIAGNOSIS_TITLES_PATH:-physionet.org/files/mimiciv/3.1/hosp/d_icd_diagnoses.csv.gz}"
EMAR_PATH="${EMAR_PATH:-physionet.org/files/mimiciv/3.1/hosp/emar.csv.gz}"
EMAR_DETAIL_PATH="${EMAR_DETAIL_PATH:-physionet.org/files/mimiciv/3.1/hosp/emar_detail.csv.gz}"
PHARMACY_PATH="${PHARMACY_PATH:-physionet.org/files/mimiciv/3.1/hosp/pharmacy.csv.gz}"
PRESCRIPTIONS_PATH="${PRESCRIPTIONS_PATH:-physionet.org/files/mimiciv/3.1/hosp/prescriptions.csv.gz}"
POE_DETAIL_PATH="${POE_DETAIL_PATH:-physionet.org/files/mimiciv/3.1/hosp/poe_detail.csv.gz}"
ICU_PROCEDUREEVENTS_PATH="${ICU_PROCEDUREEVENTS_PATH:-physionet.org/files/mimiciv/3.1/icu/procedureevents.csv.gz}"
ICU_D_ITEMS_PATH="${ICU_D_ITEMS_PATH:-physionet.org/files/mimiciv/3.1/icu/d_items.csv.gz}"
IMAGE_ROOT="${IMAGE_ROOT:-${FINAL_DIR}/data/mimic-cxr-jpg}"
CHUNKSIZE="${CHUNKSIZE:-250000}"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: ${FINAL_DIR}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR:-<script default>}"
echo "OUTPUT_MANIFEST=${OUTPUT_MANIFEST:-<none>}"
echo "CHUNKSIZE=${CHUNKSIZE}"

if command -v module >/dev/null 2>&1; then
  module load pytorch/2.4.1
fi
export PYTHONUNBUFFERED=1

cmd=(
  python -u
  "${FINAL_DIR}/src/data_preparation/build_target_trial_pairs.py"
  --config "${CONFIG_PATH}"
  --metadata-path "${METADATA_PATH}"
  --patients-path "${PATIENTS_PATH}"
  --admissions-path "${ADMISSIONS_PATH}"
  --diagnoses-path "${DIAGNOSES_PATH}"
  --diagnosis-titles-path "${DIAGNOSIS_TITLES_PATH}"
  --emar-path "${EMAR_PATH}"
  --emar-detail-path "${EMAR_DETAIL_PATH}"
  --pharmacy-path "${PHARMACY_PATH}"
  --prescriptions-path "${PRESCRIPTIONS_PATH}"
  --poe-detail-path "${POE_DETAIL_PATH}"
  --icu-procedureevents-path "${ICU_PROCEDUREEVENTS_PATH}"
  --icu-d-items-path "${ICU_D_ITEMS_PATH}"
  --image-root "${IMAGE_ROOT}"
  --chunksize "${CHUNKSIZE}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${OUTPUT_MANIFEST}" ]]; then
  cmd+=(--output-manifest "${OUTPUT_MANIFEST}")
fi

echo
echo "=== Stage 1: build target-trial cohort ==="
"${cmd[@]}"

echo
echo "Job finished at: $(date)"
