#!/bin/bash
#SBATCH --job-name=u2t-demo
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/demo-%j.out
#SBATCH --error=logs/demo-%j.err

# One-shot inference: feed a .nii.gz to a trained ckpt via demo.py.
# Override MODEL / IMAGE / QUESTION via env.

set -euo pipefail
cd /lus/lfs1aip2/projects/u6hx/huanan/u2Tokenizer
source script/slurm/_env.sh

export MODEL=${MODEL:-AlpachinoNLP/u2Qwen3-1.7B-Instruct}
export IMAGE=${IMAGE:-example.nii.gz}
export QUESTION=${QUESTION:-"Please provide a medical analysis of this image."}

python demo.py \
    --model "$MODEL" \
    --image-path "$IMAGE" \
    --question "$QUESTION"
