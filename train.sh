#!/bin/bash
# Launch training on your current Slurm GPU allocation (from salloc / GPU Assistant).
#
# Usage (from login node with salloc):
#   srun --ntasks=1 --gpus-per-task=4 bash train_slurm.sh
#
# Or from an already-allocated compute node:
#   bash train_slurm.sh
#
# Optional Hydra overrides:
#   bash train_slurm.sh training.fast_dev_run=true

set -euo pipefail

REPO_ROOT="/dss/dsshome1/0C/ra85muk2/Desktop/Programming/hindi_llama_omni"
cd "${REPO_ROOT}"

OUTPUT_DIR="outputs/training_lora"
for arg in "$@"; do
  if [[ "${arg}" == logging.output_dir=* ]]; then
    OUTPUT_DIR="${arg#logging.output_dir=}"
  fi
done

if [[ "${OUTPUT_DIR}" = /* ]]; then
  LOG_DIR="${OUTPUT_DIR}/logs"
else
  LOG_DIR="${REPO_ROOT}/${OUTPUT_DIR}/logs"
fi
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/log.err") 2>&1

# Load CUDA runtime libraries (required on HPC clusters)
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

source .venv/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# Force CUDA to reinitialize cleanly in this process
unset CUDA_LAUNCH_BLOCKING
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python omni_speech/trainer.py \
  training.accelerator=gpu \
  training.devices="${NUM_GPUS}" \
  training.strategy="${STRATEGY}" \
  training.precision=bf16-mixed \
  training.batch_size=1 \
  training.gradient_accumulation_steps=16 \
  data.num_workers=4 \
  logging.output_dir=outputs/training_lora \
  "$@"
