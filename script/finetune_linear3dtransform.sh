#!/bin/bash

# run "accelerate config" first!
export WANDB_API_KEY=00da6485031077ad0ca743ecf911ade54986ffaa
export PROJECT_PATH=/import/c4dm-04/siyoul/Med3DLLM
export CHECKPOINT_NAME=amosmm_chatgpt_phi2_l3dt_lora_0212@bs1_acc1_ep16_lr2e5_ws2_fused

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file $PROJECT_PATH/config/accelerate_config.yaml\
    --main_process_port 29501 \
    src/train/train.py \
    --version v0 \
    --model_name_or_path  $PROJECT_PATH/pretrained_models/RadPhi-2 \
    --model_type phi \
    --lora_enable True \
    --vision_tower vit3d \
    --pretrain_vision_model $PROJECT_PATH/pretrained_models/M3D-CLIP/pretrained_ViT.bin \
    --tune_mm_mlp_adapter False \
    --bf16 True \
    --train_base_path $PROJECT_PATH/datasets \
    --train_jsonl_path $PROJECT_PATH/datasets/Fused_Dataset/train/amos_mm_rewrite_chatgpt_4o_mini.jsonl \
    --val_base_path $PROJECT_PATH/datasets \
    --val_jsonl_path $PROJECT_PATH/datasets/Fused_Dataset/val/amos_mm_findings.jsonl \
    --output_dir $PROJECT_PATH/checkpoint/$CHECKPOINT_NAME \
    --num_train_epochs 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.95 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing True \
    --dataloader_pin_memory False\
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --wandb_project_name AMOS-MM \
    --wandb_run_name $CHECKPOINT_NAME \
    --freeze_vision_tower False \
    --freeze_backbone True \
    --model_max_length 1024 \
    --enable_linear_3d_tokenizer True \
