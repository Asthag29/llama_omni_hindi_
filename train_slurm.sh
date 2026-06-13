#!/bin/bash
#SBATCH --job-name=omni-train
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=cc-gpu01-n02
#SBATCH --gres=gpu:TITANRTX:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/math/gupta/work/hindi_llama_omni/LLaMA-Omni}"
CONDA_ENV="/home/groups/ai/gupta/anaconda/anaconda3/envs/llama-omni"
PYTHON="${CONDA_ENV}/bin/python"

cd "${REPO_ROOT}"
mkdir -p logs

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON}" omni_speech/trainer.py \
  training.accelerator=gpu \
  training.devices=1 \
  training.strategy=auto \
  training.precision=16-mixed \
  training.batch_size=1 \
  training.gradient_accumulation_steps=16 \
  data.num_workers=4 \
  logging.output_dir=outputs/training_1gpu_titan
