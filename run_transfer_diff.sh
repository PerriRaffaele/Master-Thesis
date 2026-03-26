python transfer_diff.py \
    --base_model_path "unsloth/Qwen2.5-Coder-1.5B" \
    --instruct_model_path "unsloth/Qwen2.5-Coder-1.5B-Instruct" \
    --finetuned_model_path "./checkpoints_15_no_lora_pl_only/checkpoint-4473" \
    --checkpoint_diff_output_path "./checkpoints_15_no_lora_pl_only/Qwen2.5-Coder-1.5B-Instruct-Continuous_9" \
    --is_peft False