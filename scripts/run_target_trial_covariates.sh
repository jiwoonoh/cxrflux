#!/bin/bash
#SBATCH --job-name=tt_cov
#SBATCH --account=ntrayan1_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=tt_cov_%j.out

set -euo pipefail

FINAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FINAL_DIR}"

CONFIG_PATH="${CONFIG_PATH:-${FINAL_DIR}/configs/primary_h24.yaml}"
PAIRS_PATH="${PAIRS_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-}"
PATIENTS_PATH="${PATIENTS_PATH:-physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz}"
ADMISSIONS_PATH="${ADMISSIONS_PATH:-physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz}"
SERVICES_PATH="${SERVICES_PATH:-physionet.org/files/mimiciv/3.1/hosp/services.csv.gz}"
ICUSTAYS_PATH="${ICUSTAYS_PATH:-physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz}"
DIAGNOSES_PATH="${DIAGNOSES_PATH:-physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz}"
DIAGNOSIS_TITLES_PATH="${DIAGNOSIS_TITLES_PATH:-physionet.org/files/mimiciv/3.1/hosp/d_icd_diagnoses.csv.gz}"
ADMIN_EVENTS_PATH="${ADMIN_EVENTS_PATH:-${FINAL_DIR}/data/cxr_pairs_frontal_iv_furosemide_heart_failure_admin_events.csv}"
LABITEMS_PATH="${LABITEMS_PATH:-physionet.org/files/mimiciv/3.1/hosp/d_labitems.csv.gz}"
LABEVENTS_PATH="${LABEVENTS_PATH:-physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz}"
OMR_PATH="${OMR_PATH:-physionet.org/files/mimiciv/3.1/hosp/omr.csv.gz}"
PRIOR_FURO_LOOKBACK_HOURS="${PRIOR_FURO_LOOKBACK_HOURS:-72}"
LAB_LOOKBACK_HOURS="${LAB_LOOKBACK_HOURS:-24}"
OMR_LOOKBACK_DAYS="${OMR_LOOKBACK_DAYS:-365}"
CHUNKSIZE="${CHUNKSIZE:-250000}"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: ${FINAL_DIR}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "PAIRS_PATH=${PAIRS_PATH:-<infer from study output dir>}"
echo "OUTPUT_DIR=${OUTPUT_DIR:-<study output dir>}"
echo "OUTPUT_MANIFEST=${OUTPUT_MANIFEST:-<none>}"
echo "CHUNKSIZE=${CHUNKSIZE}"

if command -v module >/dev/null 2>&1; then
  module load pytorch/2.4.1
fi
export PYTHONUNBUFFERED=1

cmd=(
  python -u
  "${FINAL_DIR}/src/data_preparation/build_target_trial_covariates.py"
  --config "${CONFIG_PATH}"
  --patients-path "${PATIENTS_PATH}"
  --admissions-path "${ADMISSIONS_PATH}"
  --services-path "${SERVICES_PATH}"
  --icustays-path "${ICUSTAYS_PATH}"
  --diagnoses-path "${DIAGNOSES_PATH}"
  --diagnosis-titles-path "${DIAGNOSIS_TITLES_PATH}"
  --admin-events-path "${ADMIN_EVENTS_PATH}"
  --labitems-path "${LABITEMS_PATH}"
  --labevents-path "${LABEVENTS_PATH}"
  --omr-path "${OMR_PATH}"
  --prior-furo-lookback-hours "${PRIOR_FURO_LOOKBACK_HOURS}"
  --lab-lookback-hours "${LAB_LOOKBACK_HOURS}"
  --omr-lookback-days "${OMR_LOOKBACK_DAYS}"
  --chunksize "${CHUNKSIZE}"
)

if [[ -n "${PAIRS_PATH}" ]]; then
  cmd+=(--pairs-path "${PAIRS_PATH}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${OUTPUT_MANIFEST}" ]]; then
  cmd+=(--output-manifest "${OUTPUT_MANIFEST}")
fi

echo
echo "=== Stage 3: build target-trial EHR covariates ==="
"${cmd[@]}"

echo
echo "Job finished at: $(date)"
