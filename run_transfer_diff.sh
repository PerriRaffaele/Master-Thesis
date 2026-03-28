python transfer_diff.py \
    --base_model_path "unsloth/Qwen2.5-Coder-1.5B" \
    --instruct_model_path "unsloth/Qwen2.5-Coder-1.5B-Instruct" \
    --finetuned_model_path "./checkpoints_no_lora_new/checkpoint-910" \
    --checkpoint_diff_output_path "./checkpoints_no_lora_new/Qwen2.5-Coder-1.5B-Instruct-Continuous_10" \
    --is_peft False