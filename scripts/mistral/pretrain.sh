#!/bin/bash
PRETRAIN_NAME=Mini-Vistral-2B-Pretrain
FINETUNE_NAME=Mini-Vistral-2B
AUX_SIZE=768

WANDB__SERVICE_WAIT=300 deepspeed minigemini/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path model_zoo/LLM/Vistral-7B-Chat \
    --version vistral-it \
    --data_path ./data/MiniGemini-Pretrain/data_pretrain_2.json \
    --image_folder ./data/MiniGemini-Pretrain \
    --vision_tower model_zoo/OpenAI/clip-vit-large-patch14-336 \
    --vision_tower_aux model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_size_aux $AUX_SIZE \
    --bf16 True \
    --output_dir ./work_dirs/$PRETRAIN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to wandb

