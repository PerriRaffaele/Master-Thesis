python transfer_diff.py \
    --base_model_path "unsloth/Qwen2.5-Coder-1.5B" \
    --instruct_model_path "unsloth/Qwen2.5-Coder-1.5B-Instruct" \
    --finetuned_model_path "./checkpoints_15/checkpoint-2715" \
    --checkpoint_diff_output_path "./checkpoints_15/Qwen2.5-Coder-1.5B-Instruct-Continuous" \
    --is_peft True