#!/bin/bash
#$ -l h_rt=64:0:0
#$ -l h_vmem=20G
#$ -pe smp 24
#$ -l gpu=2
#$ -l gpu_type=ampere
#$ -l gpuhighmem
#$ -cwd
#$ -j y
#$ -N amos_stage2
#$ -l rocky
#$ -l node_type=rdg
set -e

bash
module load miniforge
module load cuda/11.8.0-gcc-12.2.0

source activate med3dllm
cd /data/home/qp24681/u2Tokenizer
export WANDB_API_KEY=00da6485031077ad0ca743ecf911ade54986ffaa
export PROJECT_PATH=/data/home/qp24681/u2Tokenizer
export CHECKPOINT_NAME=amosmm_stage_2_linvt0217@bs1_acc1_ep4_lr4e6_ws4_fused

TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $PROJECT_PATH/config/accelerate_config.yaml\
    --main_process_port 29501 \
    src/train/train_stage2.py \
    --version v0 \
    --model_name_or_path  $PROJECT_PATH/sync_models/temp_models/amosmm_chatgpt_llama3.2_1b_linvt_stage1_0223@bs1_acc1_ep3_lr2e5_ws4_fused/checkpoint-18000\
    --bf16 True \
    --train_base_path /data/scratch/qp24681/datasets \
    --train_jsonl_path $PROJECT_PATH/datasets/DPO/amos_m3d_greened_merged_stage2.jsonl \
    --val_base_path /data/scratch/qp24681/datasets \
    --val_jsonl_path $PROJECT_PATH/datasets/DPO/amos_m3d_greened_merged_stage2.jsonl \
    --output_dir $PROJECT_PATH/checkpoint/$CHECKPOINT_NAME \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.95 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 5 \
    --learning_rate 4e-6 \
    --weight_decay 0. \
    --warmup_num_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing True \
    --dataloader_pin_memory True\
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --wandb_project_name AMOS-MM \
    --wandb_run_name $CHECKPOINT_NAME \
    --model_max_length 1024 \
    --enable_u2tokenizer True \
