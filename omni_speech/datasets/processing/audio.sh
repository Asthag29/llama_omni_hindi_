#!/bin/bash
#SBATCH --job-name=hindi-tts-user
#SBATCH --partition=all
#SBATCH --nodelist=cc-gpu-n[03-04]
#SBATCH --array=0-6
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tts_user_%A_%a.out
#SBATCH --error=logs/tts_user_%A_%a.err
set -euo pipefail
NUM_SHARDS=7
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/math/gupta/work/hindi_llama_omni/LLaMA-Omni}"
PYTHON="/home/groups/ai/gupta/anaconda/anaconda3/envs/llama-omni/bin/python"

cd "${REPO_ROOT}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Nodelist: ${SLURM_JOB_NODELIST}"
echo "Gres: ${SLURM_JOB_GPUS:-unset}"
echo "Cpus per task: ${SLURM_CPUS_PER_TASK}"
echo "Mem: ${SLURM_MEM_PER_NODE:-unset}"

# Data path and FLAC output path come from config.yaml and text_to_audio.py.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="/home/groups/ai/gupta/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

"${PYTHON}" omni_speech/datasets/processing/text_to_audio.py \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}" \
    --device cuda \
    --skip-existing

echo "Shard ${SHARD_ID} done at $(date)"
