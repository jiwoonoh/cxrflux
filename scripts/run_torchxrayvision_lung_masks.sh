#!/bin/bash
#SBATCH --job-name=txv_lung_masks
#SBATCH --account=ntrayan1_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=txv_lung_masks_%j.out

set -euo pipefail

FINAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FINAL_DIR}"

CSV_PATH="${CSV_PATH:-${FINAL_DIR}/results/manifests/strict_target_trial/train.csv}"
SPLIT_DIR="${SPLIT_DIR:-${FINAL_DIR}/results/manifests/strict_target_trial}"
IMAGE_ROOT="${IMAGE_ROOT:-${FINAL_DIR}/data/mimic-cxr-jpg}"
OUTPUT_DIR="${OUTPUT_DIR:-${FINAL_DIR}/results/mask_cache/torchxrayvision_chestxdet_lung256}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
SEGMENTER_SIZE="${SEGMENTER_SIZE:-512}"
FOREGROUND_CROP="${FOREGROUND_CROP:-0}"
CROP_THRESHOLD="${CROP_THRESHOLD:-10}"
CROP_MIN_CONTENT_FRACTION="${CROP_MIN_CONTENT_FRACTION:-0.02}"
CROP_MARGIN_FRACTION="${CROP_MARGIN_FRACTION:-0.03}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
INCLUDE_FOLLOWUP="${INCLUDE_FOLLOWUP:-0}"
OVERWRITE="${OVERWRITE:-0}"
MAX_IMAGES="${MAX_IMAGES:-}"

echo "Job started at: $(date)"
echo "CSV_PATH=${CSV_PATH}"
echo "SPLIT_DIR=${SPLIT_DIR}"
echo "IMAGE_ROOT=${IMAGE_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "IMAGE_SIZE=${IMAGE_SIZE}"
echo "SEGMENTER_SIZE=${SEGMENTER_SIZE}"
echo "FOREGROUND_CROP=${FOREGROUND_CROP}"
echo "INCLUDE_FOLLOWUP=${INCLUDE_FOLLOWUP}"

if command -v module >/dev/null 2>&1; then
  module load pytorch/2.4.1
fi
export PYTHONUNBUFFERED=1
mkdir -p "${OUTPUT_DIR}"

cmd=(
  python -u "${FINAL_DIR}/src/build_torchxrayvision_lung_masks.py"
  --csv-path "${CSV_PATH}"
  --image-root "${IMAGE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --split-dir "${SPLIT_DIR}"
  --image-size "${IMAGE_SIZE}"
  --segmenter-size "${SEGMENTER_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --crop-threshold "${CROP_THRESHOLD}"
  --crop-min-content-fraction "${CROP_MIN_CONTENT_FRACTION}"
  --crop-margin-fraction "${CROP_MARGIN_FRACTION}"
)

if [[ "${FOREGROUND_CROP}" == "1" ]]; then
  cmd+=(--foreground-crop)
fi
if [[ "${INCLUDE_FOLLOWUP}" == "1" ]]; then
  cmd+=(--include-followup)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi
if [[ -n "${MAX_IMAGES}" ]]; then
  cmd+=(--max-images "${MAX_IMAGES}")
fi

"${cmd[@]}"

echo "Job finished at: $(date)"
