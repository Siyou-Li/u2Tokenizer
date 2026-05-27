#!/bin/bash
#SBATCH --job-name=u2t-dpo
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=logs/dpo-%j.out
#SBATCH --error=logs/dpo-%j.err

# Stage 2 DPO. Expects:
#   - $SFT_CKPT: Stage1 ckpt directory
#   - $DPO_JSONL: preference pairs (chosen / rejected) jsonl, produced by
#     script/slurm/build_dpo_pairs.sbatch.

set -euo pipefail
cd /lus/lfs1aip2/projects/u6hx/huanan/u2Tokenizer
source script/slurm/_env.sh

export RUN_NAME=${RUN_NAME:-ct_rate_mu2_qwen3_1p7b_dpo}
export SFT_CKPT=${SFT_CKPT:-$PROJECT_CKPT/ct_rate_mu2_qwen3_1p7b_sft}
export MODEL_TYPE=${MODEL_TYPE:-qwen3}
export VIT_BIN=${VIT_BIN:-$PROJECT_PRETRAINED/M3D-CLIP/pretrained_ViT.bin}
export DPO_JSONL=${DPO_JSONL:-$PROJECT_DATA/Fused_Dataset/train/ct_rate_dpo_pairs.jsonl}
export VAL_JSONL=${VAL_JSONL:-$PROJECT_DATA/Fused_Dataset/val/amos_mm_findings.jsonl}
export DPO_BETA=${DPO_BETA:-0.1}
export NUM_EPOCHS=${NUM_EPOCHS:-1}
export PER_DEVICE_BS=${PER_DEVICE_BS:-1}
export GRAD_ACC=${GRAD_ACC:-2}
export LR=${LR:-1e-6}
export MAX_LEN=${MAX_LEN:-512}
export MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-384}
export WANDB_MODE=${WANDB_MODE:-offline}
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

NUM_GPUS=$(echo "$SLURM_GPUS_ON_NODE" | awk '{print ($0=="")?1:$0}')
echo "[dpo] GPUs=$NUM_GPUS SFT_CKPT=$SFT_CKPT"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port "$MASTER_PORT" \
    src/train/train_stage2.py \
    --version v0 \
    --model_name_or_path "$SFT_CKPT" \
    --model_type "$MODEL_TYPE" \
    --bf16 True \
    --train_base_path "$PROJECT_DATA" \
    --train_jsonl_path "$DPO_JSONL" \
    --val_base_path "$PROJECT_DATA" \
    --val_jsonl_path "$VAL_JSONL" \
    --output_dir "$PROJECT_CKPT/$RUN_NAME" \
    --beta "$DPO_BETA" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --per_device_eval_batch_size "$PER_DEVICE_BS" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate "$LR" \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 0.005 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --ddp_find_unused_parameters True \
    --dataloader_pin_memory True \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --deepspeed config/ds_config_zero2_bf16.json \
    --model_max_length "$MAX_LEN" \
    --max_length "$MAX_LEN" \
    --max_prompt_length "$MAX_PROMPT_LEN" \
    --enable_u2tokenizer True \
    --attn_type rope
