#!/bin/bash

# run "accelerate config" first!
export WANDB_API_KEY=00da6485031077ad0ca743ecf911ade54986ffaa
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29501 \
    src/train/train.py \
    --version v0 \
    --model_name_or_path /home/lez/Siyou/Med3DLLM/pretrained_models/Llama-3.2-1B-Instruct \
    --model_type llama2 \
    --vision_tower vit3d \
    --pretrain_vision_model /home/lez/Siyou/Med3DLLM/pretrained_models/M3D-CLIP/pretrained_ViT.bin \
    --tune_mm_mlp_adapter False \
    --bf16 True \
    --output_dir ./checkpoint/Med3dLLM_1116_mrg_gemma2@27_cot_bs2_acc8_ep16_lr2e5 \
    --num_train_epochs 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.95 \
    --save_strategy "steps" \
    --save_steps 688 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing True \
    --dataloader_pin_memory False\
    --dataloader_num_workers 32 \
    --report_to tensorboard \
    --wandb_project_name AMOS-MM \
    --wandb_run_name Med3dLLM_1116_mrg_gemma2@27_cot_bs2_acc8_ep16_lr2e5 \
    --freeze_vision_tower False \
    --freeze_backbone False \
    --model_max_length 512 \
    #--checkpoint_path /pfs/mt-1oY5F7/luoyihao/project/multimodal/AMOS-MM/M3D/LaMed/output/example \
