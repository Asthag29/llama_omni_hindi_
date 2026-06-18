#!/bin/bash
# Launch text-only Hindi backbone LoRA training on your current Slurm allocation.

set -euo pipefail

REPO_ROOT="/dss/dsshome1/0C/ra85muk2/Desktop/Programming/hindi_llama_omni"
cd "${REPO_ROOT}"

OUTPUT_DIR="outputs/backbone_lora"
WANDB_RUN_NAME=""
for arg in "$@"; do
  if [[ "${arg}" == logging.output_dir=* ]]; then
    OUTPUT_DIR="${arg#logging.output_dir=}"
  elif [[ "${arg}" == logging.wandb_run_name=* ]]; then
    WANDB_RUN_NAME="${arg#logging.wandb_run_name=}"
  fi
done

if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR%/}/${WANDB_RUN_ID}"
  if [[ -z "${WANDB_RUN_NAME}" ]]; then
    WANDB_RUN_NAME="backbone-sweep-${WANDB_RUN_ID}"
  fi
fi

if [[ "${OUTPUT_DIR}" = /* ]]; then
  LOG_DIR="${OUTPUT_DIR}/logs"
else
  LOG_DIR="${REPO_ROOT}/${OUTPUT_DIR}/logs"
fi
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/log.err") 2>&1

module load cuda 2>/dev/null || true

detect_num_gpus() {
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    echo "${SLURM_GPUS_ON_NODE}"
  elif [[ -n "${SLURM_GPUS:-}" ]]; then
    echo "${SLURM_GPUS}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    echo "${#_gpu_ids[@]}"
  else
    echo "1"
  fi
}

NUM_GPUS="$(detect_num_gpus)"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  STRATEGY="deepspeed_stage_3"
else
  STRATEGY="auto"
fi

echo "Using ${NUM_GPUS} GPU(s), strategy=${STRATEGY}"
echo "Output dir: ${OUTPUT_DIR}"

source .venv/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

unset CUDA_LAUNCH_BLOCKING
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PY_ARGS=(
  training.accelerator=gpu
  training.devices="${NUM_GPUS}"
  training.strategy="${STRATEGY}"
  training.precision=bf16-mixed
  data.num_workers=4
  logging.output_dir="${OUTPUT_DIR}"
)

if [[ -n "${WANDB_RUN_NAME}" ]]; then
  PY_ARGS+=("logging.wandb_run_name=${WANDB_RUN_NAME}")
fi

python omni_speech/backbone_trainer.py \
  "${PY_ARGS[@]}" \
  "$@"
