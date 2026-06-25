#!/bin/bash
#SBATCH --job-name=backbone_sweep
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="/dss/dsshome1/0C/ra85muk2/Desktop/Programming/hindi_llama_omni"
cd "${REPO_ROOT}"

SWEEP_CONFIG="${SWEEP_CONFIG:-configs/wandb_backbone_sweep.yaml}"
SWEEP_REF="${SWEEP_REF:-}"
SWEEP_AGENTS="${SWEEP_AGENTS:-1}"
GPUS_PER_AGENT="${GPUS_PER_AGENT:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agents)
      SWEEP_AGENTS="$2"
      shift 2
      ;;
    --gpus-per-agent)
      GPUS_PER_AGENT="$2"
      shift 2
      ;;
    *)
      SWEEP_REF="$1"
      shift
      ;;
  esac
done

source .venv/bin/activate

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if [[ -z "${SWEEP_REF}" ]]; then
    echo "Creating W&B sweep from ${SWEEP_CONFIG}"
    SWEEP_OUTPUT="$(wandb sweep "${SWEEP_CONFIG}" 2>&1)"
    printf '%s\n' "${SWEEP_OUTPUT}"
    SWEEP_REF="$(printf '%s\n' "${SWEEP_OUTPUT}" | sed -n 's/.*wandb agent //p' | tail -n 1)"
  fi

  if [[ -z "${SWEEP_REF}" ]]; then
    echo "Could not determine sweep id/path."
    exit 1
  fi

  mkdir -p logs/slurm
  ARRAY_END=$((SWEEP_AGENTS - 1))
  echo "Submitting ${SWEEP_AGENTS} Slurm sweep agent(s), ${GPUS_PER_AGENT} GPU(s) each, for ${SWEEP_REF}"
  sbatch \
    --array="0-${ARRAY_END}" \
    --gres="gpu:${GPUS_PER_AGENT}" \
    --export=ALL,SWEEP_REF="${SWEEP_REF}" \
    "$0"
  exit 0
fi

if [[ -z "${SWEEP_REF}" ]]; then
  echo "SWEEP_REF is required inside Slurm jobs."
  exit 1
fi

bash run_wandb_backbone_sweep.sh "${SWEEP_REF}"