#!/bin/bash
#SBATCH --job-name=hindi-tts-user
#SBATCH --partition=all
#SBATCH --nodelist=cc-gpu01-n02
#SBATCH --array=0-2
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tts_user_%A_%a.out
#SBATCH --error=logs/tts_user_%A_%a.err
echo "Printing info"
print_info() {
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
    echo "Nodelist: ${SLURM_JOB_NODELIST}"
    echo "Gres: ${SLURM_JOB_GPUS}"
    echo "Cpus per task: ${SLURM_CPUS_PER_TASK}"
    echo "Mem: ${SLURM_MEM_PER_NODE}"
}
#! each job use 1 gpu that's why have gres:1
set -euo pipefail
NUM_SHARDS=4
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/math/gupta/work/hindi_llama_omni/LLaMA-Omni}"
cd "${REPO_ROOT}"
PYTHON="/home/groups/ai/gupta/anaconda/anaconda3/envs/llama-omni/bin/python"

cd "${REPO_ROOT}"


# Output paths (wav/, manifest*.json) come from config.yaml → data.path
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="/home/groups/ai/gupta/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

"${PYTHON}" omni_speech/datasets/processing/text_to_audio.py \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}" \
    --device cuda \
    --skip-existing

echo "Shard ${SHARD_ID} done at $(date)"
