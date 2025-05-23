export WANDB_API_KEY=00da6485031077ad0ca743ecf911ade54986ffaa
export PROJECT_PATH=/data/home/qp24681/u2Tokenizer
export CHECKPOINT_NAME=amos_mm_final/amos_mm_linvt@bs1_acc1_ep4_lr4e6_ws4_fused
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
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.95 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 4e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing True \
    --dataloader_pin_memory True\
    --dataloader_num_workers 12 \
    --report_to tensorboard \
    --wandb_project_name AMOS-MM \
    --wandb_run_name $CHECKPOINT_NAME \
    --freeze_vision_tower False \
    --freeze_backbone False \
    --model_max_length 1024 \
    --enable_u2tokenizer True \
    --enable_rpe False \
    --enable_diffts False \
    --enable_dmtp False \
