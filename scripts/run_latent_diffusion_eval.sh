#!/bin/bash
#SBATCH --job-name=v2_eval
#SBATCH --account=ntrayan1_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=v2_eval_%j.out

set -euo pipefail

FINAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FINAL_DIR}"

CHECKPOINT="${CHECKPOINT:-${FINAL_DIR}/checkpoints/v2_mean_residual/best_model.pt}"
CSV_PATH="${CSV_PATH:-${FINAL_DIR}/results/manifests/strict_target_trial/train.csv}"
SPLIT_DIR="${SPLIT_DIR:-${FINAL_DIR}/results/manifests/strict_target_trial}"
IMAGE_ROOT="${IMAGE_ROOT:-${FINAL_DIR}/data/mimic-cxr-jpg}"
OUTPUT_DIR="${OUTPUT_DIR:-${FINAL_DIR}/results/runs/v2_mean_residual_eval}"
SPLIT="${SPLIT:-test}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAMPLE_COUNT="${SAMPLE_COUNT:-8}"
SAMPLE_START_TIMESTEP="${SAMPLE_START_TIMESTEP:-100}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SEED="${SEED:-42}"
DISABLE_AMP="${DISABLE_AMP:-0}"
SKIP_IMAGE_VERIFY="${SKIP_IMAGE_VERIFY:-0}"

if command -v module >/dev/null 2>&1; then
  module load pytorch/2.4.1
fi
export PYTHONUNBUFFERED=1
mkdir -p "${OUTPUT_DIR}"

cmd=(
  python -u "${FINAL_DIR}/src/evaluate_latent_diffusion.py"
  --checkpoint "${CHECKPOINT}"
  --csv-path "${CSV_PATH}"
  --image-root "${IMAGE_ROOT}"
  --split-dir "${SPLIT_DIR}"
  --split "${SPLIT}"
  --output-dir "${OUTPUT_DIR}"
  --image-size "${IMAGE_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --sample-count "${SAMPLE_COUNT}"
  --sample-start-timestep "${SAMPLE_START_TIMESTEP}"
  --sample-steps "${SAMPLE_STEPS}"
  --seed "${SEED}"
)

if [[ "${DISABLE_AMP}" == "1" ]]; then
  cmd+=(--disable-amp)
fi
if [[ "${SKIP_IMAGE_VERIFY}" == "1" ]]; then
  cmd+=(--skip-image-verify)
fi

"${cmd[@]}"
