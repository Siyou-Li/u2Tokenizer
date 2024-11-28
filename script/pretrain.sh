#!/bin/bash

# run "accelerate config" first!
export WANDB_API_KEY=00da6485031077ad0ca743ecf911ade54986ffaa
export PROJECT_PATH=/pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/Med3D_LLM
export CHECKPOINT_NAME=Med3dLLM_1128_mrg_qwen2.5@72b_cot_bs8_acc1_ep16_lr2e5_ws4_fused

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $PROJECT_PATH/config/accelerate_config.json\
    --main_process_port 29501 \
    src/train/train.py \
    --version v0 \
    --model_name_or_path  $PROJECT_PATH/pretrained_models/Llama-3.2-1B-Instruct \
    --model_type llama3.2 \
    --vision_tower vit3d \
    --pretrain_vision_model $PROJECT_PATH/pretrained_models/M3D-CLIP/pretrained_ViT.bin \
    --tune_mm_mlp_adapter False \
    --bf16 True \
    --train_base_path $PROJECT_PATH/datasets \
    --train_jsonl_path $PROJECT_PATH/datasets/Fused_Dataset/fused_train_dataset.jsonl \
    --val_image_dir $PROJECT_PATH/datasets/AMOS-MM/ \
    --val_json_path $PROJECT_PATH/datasets/AMOS-MM/dataset_{}.json \
    --output_dir $PROJECT_PATH/checkpoint/$CHECKPOINT_NAME \
    --num_train_epochs 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.95 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing True \
    --dataloader_pin_memory False\
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --wandb_project_name AMOS-MM \
    --wandb_run_name $CHECKPOINT_NAME \
    --freeze_vision_tower False \
    --freeze_backbone False \
    --model_max_length 2048 \
