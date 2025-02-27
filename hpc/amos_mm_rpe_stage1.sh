#!/bin/bash
#$ -l h_rt=64:0:0
#$ -l h_vmem=20G
#$ -pe smp 24
#$ -l gpu=2
#$ -l gpu_type=ampere
#$ -l gpuhighmem
#$ -cwd
#$ -j y
#$ -N amos_mm_rpe_stage1
#$ -l rocky
#$ -l node_type=rdg
set -e

bash
module load miniforge
module load cuda/11.8.0-gcc-12.2.0

source activate med3dllm
cd /data/home/qp24681/Med3DLLM
export WANDB_API_KEY=00da6485031077ad0ca743ecf911ade54986ffaa
export PROJECT_PATH=/data/home/qp24681/Med3DLLM
export CHECKPOINT_NAME=amos_mm_rpe_0226@bs2_acc1_ep4_lr4e6_ws2_fused
echo $SGE_HGR_gpu 
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $PROJECT_PATH/config/accelerate_config.yaml\
    --main_process_port 29501 \
    src/train/train_stage1.py \
    --version v0 \
    --model_name_or_path  $PROJECT_PATH/pretrained_models/Llama-3.2-1B-Instruct \
    --model_type llama \
    --lora_enable False \
    --vision_tower vit3d \
    --pretrain_vision_model $PROJECT_PATH/pretrained_models/M3D-CLIP/pretrained_ViT.bin \
    --tune_mm_mlp_adapter False \
    --bf16 True \
    --train_base_path /data/scratch/qp24681/datasets \
    --train_jsonl_path $PROJECT_PATH/datasets/Fused_Dataset/train/amos_mm_rewrite_chatgpt_4o_mini.jsonl \
    --val_base_path /data/scratch/qp24681/datasets \
    --val_jsonl_path $PROJECT_PATH/datasets/Fused_Dataset/val/amos_mm_findings.jsonl \
    --output_dir $PROJECT_PATH/checkpoint/$CHECKPOINT_NAME \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
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
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing True \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --wandb_project_name AMOS-MM \
    --wandb_run_name $CHECKPOINT_NAME \
    --freeze_vision_tower False \
    --freeze_backbone False \
    --model_max_length 1024 \
    --enable_linear_3d_tokenizer True \
    --enable_rpe True \
    --enable_diffts False \
    --enable_dmtp False \
