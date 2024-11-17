#!/bin/bash

# run "accelerate config" first!
export WANDB_API_KEY=00da6485031077ad0ca743ecf911ade54986ffaa
export PYTHONPATH=/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/M3D
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29501 \
    ./MedLLM/src/train/train.py \
    --version v0 \
    --model_name_or_path /pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/M3D/pretrained_model/Asclepius-Llama3-8B \
    --model_type llama2 \
    --lora_enable True \
    --vision_tower vit3d \
    --pretrain_vision_model /pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/M3D/pretrained_model/M3D-CLIP/pretrained_ViT.bin \
    --tune_mm_mlp_adapter False \
    --bf16 True \
    --output_dir ./MedLLM/output/LaMed-Asclepius8b-1001-mrg-lora \
    --num_train_epochs 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.95 \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 4 \
    --report_to wandb \
    --wandb_project_name AMOS-MM \
    --wandb_run_name Asclepius8b-1001-mrg-lora \
    --freeze_vision_tower False \
    --freeze_backbone True \
    --model_max_length 512 \
    #--checkpoint_path /pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/M3D/LaMed/output/example \
