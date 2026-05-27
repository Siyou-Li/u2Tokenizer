#!/bin/bash
#SBATCH --job-name=u2t-download
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/download-%j.out
#SBATCH --error=logs/download-%j.err

# Downloads the base LLM (Qwen3-1.7B-Instruct) + M3D-CLIP ViT into
# $PROJECT_PRETRAINED. Run AFTER setup_env.sbatch.

set -euo pipefail
cd /lus/lfs1aip2/projects/u6hx/huanan/u2Tokenizer
source script/slurm/_env.sh

python -m pip install -q "huggingface_hub[cli]"

# --- Base LLM ------------------------------------------------------------
LLM_REPO=${LLM_REPO:-Qwen/Qwen3-1.7B}
LLM_DIR="$PROJECT_PRETRAINED/$(basename "$LLM_REPO")"
if [ ! -f "$LLM_DIR/config.json" ]; then
    echo "[dl] $LLM_REPO -> $LLM_DIR"
    huggingface-cli download "$LLM_REPO" \
        --local-dir "$LLM_DIR" --local-dir-use-symlinks False
else
    echo "[dl] $LLM_DIR already populated"
fi

# --- 3D Vision Tower (M3D-CLIP) -----------------------------------------
VIT_DIR="$PROJECT_PRETRAINED/M3D-CLIP"
if [ ! -f "$VIT_DIR/pretrained_ViT.bin" ]; then
    echo "[dl] M3D-CLIP -> $VIT_DIR"
    huggingface-cli download GoodBaiBai88/M3D-CLIP \
        --local-dir "$VIT_DIR" --local-dir-use-symlinks False \
        --include "pretrained_ViT.bin" "*.json"
else
    echo "[dl] $VIT_DIR already populated"
fi

ls -lh "$LLM_DIR" "$VIT_DIR"
echo "[dl] DONE"
