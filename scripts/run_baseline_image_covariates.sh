#!/bin/bash
#SBATCH --job-name=tt_imgcov
#SBATCH --account=ntrayan1_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=tt_imgcov_%j.out

set -euo pipefail

FINAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FINAL_DIR}"

CONFIG_PATH="${CONFIG_PATH:-${FINAL_DIR}/configs/primary_h24.yaml}"
PAIRS_PATH="${PAIRS_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-}"
RESIZE="${RESIZE:-32}"
PCA_COMPONENTS="${PCA_COMPONENTS:-16}"
PROGRESS_EVERY="${PROGRESS_EVERY:-250}"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: ${FINAL_DIR}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "PAIRS_PATH=${PAIRS_PATH:-<infer from study output dir>}"
echo "OUTPUT_DIR=${OUTPUT_DIR:-<study output dir>}"
echo "OUTPUT_MANIFEST=${OUTPUT_MANIFEST:-<none>}"
echo "RESIZE=${RESIZE}"
echo "PCA_COMPONENTS=${PCA_COMPONENTS}"
echo "PROGRESS_EVERY=${PROGRESS_EVERY}"

if command -v module >/dev/null 2>&1; then
  module load pytorch/2.4.1
fi
export PYTHONUNBUFFERED=1

cmd=(
  python -u
  "${FINAL_DIR}/src/data_preparation/build_baseline_image_covariates.py"
  --config "${CONFIG_PATH}"
  --resize "${RESIZE}"
  --pca-components "${PCA_COMPONENTS}"
  --progress-every "${PROGRESS_EVERY}"
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
echo "=== Stage 2: build baseline image covariates ==="
"${cmd[@]}"

echo
echo "Job finished at: $(date)"
