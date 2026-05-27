#!/bin/bash
#SBATCH --job-name=u2t-dpo-build
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/dpo-build-%j.out
#SBATCH --error=logs/dpo-build-%j.err

# Build DPO (chosen, rejected) pairs by sampling from a Stage1 ckpt and
# scoring with a metric (rougeL by default).
#
# Overridable via env:
#   STAGE1_CKPT          path or HF id of the Stage 1 model
#   SRC_JSONL            source SFT jsonl (image/question/answer)
#   BASE_PATH            image root (joined with row['image'])
#   OUT_JSONL            destination jsonl
#   N_CANDIDATES         samples per row (default 4)
#   METRIC               rougeL | bleu | green
#   MAX_SAMPLES          cap on src rows (smoke: 8)

set -euo pipefail
cd /lus/lfs1aip2/projects/u6hx/huanan/u2Tokenizer
source script/slurm/_env.sh

export STAGE1_CKPT=${STAGE1_CKPT:-AlpachinoNLP/u2Qwen3-1.7B-Instruct}
export SRC_JSONL=${SRC_JSONL:-$PROJECT_DATA/Fused_Dataset/train/ct_rate_smoke.jsonl}
export BASE_PATH=${BASE_PATH:-$PROJECT_DATA}
export OUT_JSONL=${OUT_JSONL:-$PROJECT_DATA/Fused_Dataset/train/ct_rate_dpo_pairs.jsonl}
export N_CANDIDATES=${N_CANDIDATES:-4}
export METRIC=${METRIC:-rougeL}
export MAX_SAMPLES=${MAX_SAMPLES:-}

NUM_GPUS=$(echo "$SLURM_GPUS_ON_NODE" | awk '{print ($0=="")?1:$0}')

CMD=(python src/preprocess/build_dpo_pairs.py
    --model_path "$STAGE1_CKPT"
    --src_jsonl  "$SRC_JSONL"
    --base_path  "$BASE_PATH"
    --out_jsonl  "$OUT_JSONL"
    --n_candidates "$N_CANDIDATES"
    --metric     "$METRIC"
    --num_gpus   "$NUM_GPUS")
if [ -n "$MAX_SAMPLES" ]; then
    CMD+=(--max_samples "$MAX_SAMPLES")
fi

echo "[dpo-build] ${CMD[*]}"
"${CMD[@]}"
echo "[dpo-build] DONE -> $OUT_JSONL"
