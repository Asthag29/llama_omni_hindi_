#!/bin/bash
#SBATCH --job-name=backbone_full_ufd0nu7q
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="/dss/dsshome1/0C/ra85muk2/Desktop/Programming/hindi_llama_omni"
cd "${REPO_ROOT}"

mkdir -p logs/slurm

source .venv/bin/activate   


# Full-data backbone training using configs/backbone.yaml.
# This is a normal training job, not a W&B sweep agent.
bash train_backbone.sh
