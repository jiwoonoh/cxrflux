#!/bin/bash
#SBATCH --job-name=tt_ldm
#SBATCH --account=ntrayan1_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=tt_ldm_%j.out

set -euo pipefail

FINAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FINAL_DIR}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${FINAL_DIR}/results/runs/latent_diffusion_cxr256_v1}"
IMAGE_ROOT="${IMAGE_ROOT:-${FINAL_DIR}/data/mimic-cxr-jpg}"
AUTOENCODER_CHECKPOINT="${AUTOENCODER_CHECKPOINT:-${FINAL_DIR}/checkpoints/autoencoder/best_model.pt}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
BRIDGE_CHECKPOINT="${BRIDGE_CHECKPOINT:-}"
BRIDGE_METHOD="${BRIDGE_METHOD:-}"
BRIDGE_INFERENCE_STEPS="${BRIDGE_INFERENCE_STEPS:-0}"
TARGET_MODE="${TARGET_MODE:-mean_residual}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-1.0}"
MEAN_BASE_CHANNELS="${MEAN_BASE_CHANNELS:-64}"
MEAN_LOSS_WEIGHT="${MEAN_LOSS_WEIGHT:-1.0}"
ALLOW_PARTIAL_INIT="${ALLOW_PARTIAL_INIT:-0}"
CSV_PATH="${CSV_PATH:-${FINAL_DIR}/results/manifests/strict_target_trial/train.csv}"
SPLIT_DIR="${SPLIT_DIR:-${FINAL_DIR}/results/manifests/strict_target_trial}"
RUN_NAME="${RUN_NAME:-v2_mean_residual_retrain}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
SAVE_PATH="${SAVE_PATH:-${OUTPUT_DIR}/best_model.pt}"
SAMPLE_DIR="${SAMPLE_DIR:-${OUTPUT_DIR}/samples}"
LOSS_WEIGHT_COLUMN="${LOSS_WEIGHT_COLUMN:-balancing_weight}"

IMAGE_SIZE="${IMAGE_SIZE:-256}"
FOREGROUND_CROP="${FOREGROUND_CROP:-0}"
CROP_THRESHOLD="${CROP_THRESHOLD:-10}"
CROP_MIN_CONTENT_FRACTION="${CROP_MIN_CONTENT_FRACTION:-0.02}"
CROP_MARGIN_FRACTION="${CROP_MARGIN_FRACTION:-0.03}"
LATENT_BASE_CHANNELS="${LATENT_BASE_CHANNELS:-128}"
BATCH_SIZE="${BATCH_SIZE:-24}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-12}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
SAMPLE_START_TIMESTEP="${SAMPLE_START_TIMESTEP:-250}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SKIP_IMAGE_VERIFY="${SKIP_IMAGE_VERIFY:-0}"
DISABLE_AMP="${DISABLE_AMP:-0}"

echo "Job started at: $(date)"
echo "CSV_PATH=${CSV_PATH}"
echo "SPLIT_DIR=${SPLIT_DIR}"
echo "AUTOENCODER_CHECKPOINT=${AUTOENCODER_CHECKPOINT}"
echo "INIT_CHECKPOINT=${INIT_CHECKPOINT}"
echo "BRIDGE_CHECKPOINT=${BRIDGE_CHECKPOINT:-<none>}"
echo "TARGET_MODE=${TARGET_MODE}"
echo "RESIDUAL_SCALE=${RESIDUAL_SCALE}"
echo "MEAN_BASE_CHANNELS=${MEAN_BASE_CHANNELS}"
echo "MEAN_LOSS_WEIGHT=${MEAN_LOSS_WEIGHT}"
echo "ALLOW_PARTIAL_INIT=${ALLOW_PARTIAL_INIT}"
echo "FOREGROUND_CROP=${FOREGROUND_CROP}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

if command -v module >/dev/null 2>&1; then
  module load pytorch/2.4.1
fi
export PYTHONUNBUFFERED=1
mkdir -p "${OUTPUT_DIR}" "${SAMPLE_DIR}"

cmd=(
  python -u "${FINAL_DIR}/src/train_latent_diffusion.py"
  --csv-path "${CSV_PATH}"
  --image-root "${IMAGE_ROOT}"
  --split-dir "${SPLIT_DIR}"
  --autoencoder-checkpoint "${AUTOENCODER_CHECKPOINT}"
  --init-checkpoint "${INIT_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --save-path "${SAVE_PATH}"
  --sample-dir "${SAMPLE_DIR}"
  --image-size "${IMAGE_SIZE}"
  --latent-base-channels "${LATENT_BASE_CHANNELS}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --loss-weight-column "${LOSS_WEIGHT_COLUMN}"
  --target-mode "${TARGET_MODE}"
  --residual-scale "${RESIDUAL_SCALE}"
  --mean-base-channels "${MEAN_BASE_CHANNELS}"
  --mean-loss-weight "${MEAN_LOSS_WEIGHT}"
  --sample-start-timestep "${SAMPLE_START_TIMESTEP}"
  --sample-steps "${SAMPLE_STEPS}"
  --crop-threshold "${CROP_THRESHOLD}"
  --crop-min-content-fraction "${CROP_MIN_CONTENT_FRACTION}"
  --crop-margin-fraction "${CROP_MARGIN_FRACTION}"
)

if [[ -n "${BRIDGE_CHECKPOINT}" ]]; then
  cmd+=(--bridge-checkpoint "${BRIDGE_CHECKPOINT}")
fi
if [[ -n "${BRIDGE_METHOD}" ]]; then
  cmd+=(--bridge-method "${BRIDGE_METHOD}")
fi
if [[ "${BRIDGE_INFERENCE_STEPS}" != "0" ]]; then
  cmd+=(--bridge-inference-steps "${BRIDGE_INFERENCE_STEPS}")
fi
if [[ "${ALLOW_PARTIAL_INIT}" == "1" ]]; then
  cmd+=(--allow-partial-init)
fi
if [[ "${FOREGROUND_CROP}" == "1" ]]; then
  cmd+=(--foreground-crop)
fi
if [[ "${SKIP_IMAGE_VERIFY}" == "1" ]]; then
  cmd+=(--skip-image-verify)
fi
if [[ "${DISABLE_AMP}" == "1" ]]; then
  cmd+=(--disable-amp)
fi

"${cmd[@]}"

echo "Job finished at: $(date)"
