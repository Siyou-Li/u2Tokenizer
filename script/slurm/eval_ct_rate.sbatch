#!/bin/bash
#SBATCH --job-name=u2t-eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

# Multi-GPU CT-RATE evaluation. Tweak via env:
#   MODEL_PATH, DATA_DIR, MAX_SAMPLES, METRICS (space-separated).

set -euo pipefail
cd /lus/lfs1aip2/projects/u6hx/huanan/u2Tokenizer
source script/slurm/_env.sh

export MODEL_PATH=${MODEL_PATH:-$PROJECT_CKPT/ct_rate_mu2_qwen3_1p7b_sft}
export DATA_DIR=${DATA_DIR:-$PROJECT_DATA/CT-RATE/dataset/valid}
export CSV_PATH=${CSV_PATH:-script/evaluation/valid_labels.csv}
export MAX_SAMPLES=${MAX_SAMPLES:-50}
export METRICS=${METRICS:-bleu rouge bert meteor}   # add 'green' to enable LLM scorer
export LOG_FILE=${LOG_FILE:-logs/eval_results_${SLURM_JOB_ID}.txt}

NUM_GPUS=$(echo "$SLURM_GPUS_ON_NODE" | awk '{print ($0=="")?1:$0}')
echo "[eval] GPUs=$NUM_GPUS MODEL_PATH=$MODEL_PATH METRICS=$METRICS"

python script/evaluation/ct_rate.py \
    --model_path "$MODEL_PATH" \
    --data_dir  "$DATA_DIR" \
    --csv_path  "$CSV_PATH" \
    --num_gpus  "$NUM_GPUS" \
    --max_samples "$MAX_SAMPLES" \
    --log_file  "$LOG_FILE" \
    --metrics   $METRICS

echo "[eval] DONE -> $LOG_FILE"
