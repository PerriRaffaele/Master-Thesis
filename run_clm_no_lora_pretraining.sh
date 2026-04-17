#!/bin/bash

MODEL_NAME="unsloth/Qwen2.5-Coder-1.5B"
CODE_FILEPATH="./benchmarks/training/mceval_and_2k_multi_language.jsonl"

python -u clm_no_lora.py \
    --model_name "$MODEL_NAME" \
    --training_data "$CODE_FILEPATH" \
    --output_dir "checkpoints_with_2k_multi" \
    --epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-05 \
    --lr_scheduler_type "linear" \
    --logging_steps 5 \
    --seed 42 \