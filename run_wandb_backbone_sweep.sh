#!/bin/bash
# Wrapper for W&B sweeps: activate env and isolate each run's outputs.

set -euo pipefail

REPO_ROOT="/dss/dsshome1/0C/ra85muk2/Desktop/Programming/hindi_llama_omni"
cd "${REPO_ROOT}"

source .venv/bin/activate

RUN_TAG="${WANDB_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
OUTPUT_DIR="outputs/backbone_lora_sweep/${RUN_TAG}"
RUN_NAME="backbone-sweep-${RUN_TAG}"

bash train_backbone.sh \
  logging.output_dir="${OUTPUT_DIR}" \
  logging.wandb_run_name="${RUN_NAME}" \
  "$@"
