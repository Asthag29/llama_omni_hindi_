#!/bin/bash
#SBATCH --job-name=hindi-tts-user
#SBATCH --partition=all
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/tts_user_%A_%a.out
#SBATCH --error=logs/tts_user_%A_%a.err

set -euo pipefail
NUM_SHARDS=2
SHARD_ID="${SLURM_ARRAY_TASK_ID}"
REPO_ROOT="/home/math/gupta/work/hindi_llama_omni/LLaMA-Omni"
PYTHON="/home/groups/ai/gupta/anaconda/anaconda3/envs/llama-omni/bin/python"

cd "${REPO_ROOT}"
mkdir -p logs data/processed/hindi_wav/manifest_shards

# Set env variables explicitly, do NOT source bashrc
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="/home/groups/ai/gupta/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# Use full python path, no conda activate needed
"${PYTHON}" omni_speech/datasets/processing/text_to_audio.py \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}" \
    --device cuda \
    --skip-existing

echo "Shard ${SHARD_ID} done at $(date)"