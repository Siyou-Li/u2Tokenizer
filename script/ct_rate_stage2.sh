#!/bin/bash

# run "accelerate config" first!
export WANDB_API_KEY=00da6485031077ad0ca743ecf911ade54986ffaa
export PROJECT_PATH=/import/c4dm-04/siyoul/u2Tokenizer
export CHECKPOINT_NAME=ct_rate_stage2_linvt_0227@bs1_acc1_ep6_lr4e6_ws4_fused

TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $PROJECT_PATH/config/accelerate_config.yaml\
    --main_process_port 29501 \
    src/train/train_stage2.py \
    --version v0 \
    --model_name_or_path  $PROJECT_PATH/sync_models/ct_rate_llama3.2_1b_linvt_stage1_0224@bs1_acc1_ep4_lr4e6_ws4_fused/checkpoint-26000\
    --bf16 True \
    --train_base_path $PROJECT_PATH/datasets \
    --train_jsonl_path $PROJECT_PATH/datasets/DPO/ct_rate_greened_merged_stage2.jsonl \
    --val_base_path $PROJECT_PATH/datasets \
    --val_jsonl_path $PROJECT_PATH/datasets/DPO/ct_rate_greened_merged_stage2.jsonl \
    --output_dir $PROJECT_PATH/checkpoint/$CHECKPOINT_NAME \
    --num_train_epochs 6 \
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
