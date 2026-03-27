#!/bin/bash

MODEL_NAME="unsloth/Qwen2.5-Coder-1.5B"
CODE_FILEPATH="./benchmarks/training/combined_traning_data.jsonl"

python -u clm_no_lora.py \
    --model_name "$MODEL_NAME" \
    --training_data "$CODE_FILEPATH" \
    --output_dir "checkpoints_no_lora" \
    --epochs 15 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-05 \
    --lr_scheduler_type "linear" \
    --logging_steps 5 \
    --seed 42 \