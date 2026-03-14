#!/bin/bash

MODEL_NAME="unsloth/Qwen2.5-Coder-1.5B"
CODE_FILEPATH="./benchmarks/training/combined_training_data.jsonl"

python3 -u clm.py \
    --model_name "$MODEL_NAME" \
    --code_column "content" \
    --hard_code_qwen \
    --max_model_length 2048 \
    --training_data "$CODE_FILEPATH" \
    --output_dir "checkpoints" \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --batch_size 1 \
    --epochs 5 \
    --gradient_accumulation_steps 4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --save_total_limit 100 \
    --report_to "none" \
    --num_proc 64