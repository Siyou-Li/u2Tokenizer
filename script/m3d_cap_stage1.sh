#!/bin/bash

# run "accelerate config" first!
export WANDB_API_KEY=00da6485031077ad0ca743ecf911ade54986ffaa
export PROJECT_PATH=/import/c4dm-04/siyoul/u2Tokenizer
export CHECKPOINT_NAME=m3d_cap_u2t_0214@bs4_acc1_ep8_lr2e5_ws4

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $PROJECT_PATH/config/accelerate_config.yaml\
    --main_process_port 29501 \
    src/train/med3d_llm_train.py \
    --version v0 \
    --model_name_or_path  $PROJECT_PATH/pretrained_models/M3D-u2-Phi-3-4B \
    --model_type phi3 \
    --lora_enable False \
    --vision_tower vit3d \
    --pretrain_vision_model $PROJECT_PATH/pretrained_models/M3D-CLIP/pretrained_ViT.bin \
    --pretrain_mm_mlp_adapter $PROJECT_PATH/pretrained_models/M3D-u2-Phi-3-4B/mm_projector.bin \
    --tune_mm_mlp_adapter False \
    --bf16 True \
    --output_dir $PROJECT_PATH/checkpoint/$CHECKPOINT_NAME \
    --num_train_epochs 8 \
    --per_device_train_batch_size 4 \
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
    --freeze_vision_tower True \
    --freeze_backbone True \
    --model_max_length 1024 \
    --enable_u2tokenizer True \
