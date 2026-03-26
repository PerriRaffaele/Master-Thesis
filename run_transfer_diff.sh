python transfer_diff.py \
    --base_model_path "unsloth/Qwen2.5-Coder-1.5B" \
    --instruct_model_path "unsloth/Qwen2.5-Coder-1.5B-Instruct" \
    --finetuned_model_path "./checkpoints_no_lora/checkpoint-4344" \
    --checkpoint_diff_output_path "./checkpoints_no_lora/Qwen2.5-Coder-1.5B-Instruct-Continuous_6" \
    --is_peft False