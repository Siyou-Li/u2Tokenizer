#!/bin/bash
#SBATCH --job-name=u2t-sft
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=logs/sft-%j.out
#SBATCH --error=logs/sft-%j.err

# Stage 1 SFT — Qwen3-1.7B base. Adapted from script/qwen3-1.7B-ft.sh.
# Overridable via env vars (CHECKPOINT_NAME, MODEL_NAME, etc).

set -euo pipefail
cd /lus/lfs1aip2/projects/u6hx/huanan/u2Tokenizer
source script/slurm/_env.sh

export CHECKPOINT_NAME=${CHECKPOINT_NAME:-ct_rate_mu2_qwen3_1p7b_sft}
export MODEL_NAME=${MODEL_NAME:-$PROJECT_PRETRAINED/Qwen3-1.7B}
export MODEL_TYPE=${MODEL_TYPE:-qwen3}
export VIT_BIN=${VIT_BIN:-$PROJECT_PRETRAINED/M3D-CLIP/pretrained_ViT.bin}
export TRAIN_JSONL=${TRAIN_JSONL:-$PROJECT_DATA/Fused_Dataset/train/ct_rate_raw.jsonl}
export VAL_JSONL=${VAL_JSONL:-$PROJECT_DATA/Fused_Dataset/val/amos_mm_findings.jsonl}
export NUM_EPOCHS=${NUM_EPOCHS:-4}
export MAX_STEPS=${MAX_STEPS:--1}      # >0 overrides NUM_EPOCHS for smoke runs
export PER_DEVICE_BS=${PER_DEVICE_BS:-1}
export GRAD_ACC=${GRAD_ACC:-1}
export LR=${LR:-4e-6}
export MAX_LEN=${MAX_LEN:-1024}
export SAVE_STEPS=${SAVE_STEPS:-2000}
export EVAL_STEPS=${EVAL_STEPS:-0.95}

export WANDB_MODE=${WANDB_MODE:-offline}

# Master port — derive from job id to avoid collisions when running multiple
# trainings concurrently.
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

NUM_GPUS=$(echo "$SLURM_GPUS_ON_NODE" | awk '{print ($0=="")?1:$0}')
echo "[sft] GPUs=$NUM_GPUS MODEL=$MODEL_NAME"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port "$MASTER_PORT" \
    src/train/train_stage1.py \
    --version v0 \
    --model_name_or_path "$MODEL_NAME" \
    --model_type "$MODEL_TYPE" \
    --lora_enable False \
    --vision_tower vit3d \
    --pretrain_vision_model "$VIT_BIN" \
    --tune_mm_mlp_adapter False \
    --bf16 True \
    --train_base_path "$PROJECT_DATA" \
    --train_jsonl_path "$TRAIN_JSONL" \
    --val_base_path "$PROJECT_DATA" \
    --val_jsonl_path "$VAL_JSONL" \
    --output_dir "$PROJECT_CKPT/$CHECKPOINT_NAME" \
    --num_train_epochs "$NUM_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --per_device_eval_batch_size "$PER_DEVICE_BS" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --evaluation_strategy steps \
    --eval_accumulation_steps 1 \
    --eval_steps "$EVAL_STEPS" \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 5 \
    --learning_rate "$LR" \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 0.001 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --ddp_find_unused_parameters True \
    --dataloader_pin_memory True \
    --dataloader_num_workers 10 \
    --report_to tensorboard \
    --wandb_project_name u2Tokenizer \
    --wandb_run_name "$CHECKPOINT_NAME" \
    --freeze_vision_tower False \
    --freeze_backbone False \
    --model_max_length "$MAX_LEN" \
    --enable_u2tokenizer True \
    --attn_type rope \
    --enable_diffts True \
    --enable_dmtp True
