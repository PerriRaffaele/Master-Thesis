# python transfer_diff.py \
#     --base_model_path "unsloth/Qwen2.5-Coder-1.5B" \
#     --instruct_model_path "unsloth/Qwen2.5-Coder-1.5B-Instruct" \
#     --finetuned_model_path "Devy1/Qwen2.5-Coder-CONTROL-checkpoints_python_only_2k-1.5B-Base-6" \
#     --checkpoint_diff_output_path "./checkpoints_python_only_2k/Qwen2.5-Coder-1.5B-Instruct-Continuous_6" \
#     --is_peft False

for EPOCHS in {1..10}; do
    echo "====================================================="
    echo "Starting transfer_diff for Epoch: $EPOCHS"
    echo "====================================================="

    # Dynamically inject the current epoch number into the paths
    HF_PATH="Devy1/Qwen2.5-Coder-CONTROL-checkpoints_multi_language_2k-1.5B-Base-${EPOCHS}"
    OUTPUT_PATH="./checkpoints_multi_language_2k/Qwen2.5-Coder-1.5B-Instruct-Continuous_${EPOCHS}"

    # Run the python script with the dynamic paths
    python transfer_diff.py \
        --base_model_path "unsloth/Qwen2.5-Coder-1.5B" \
        --instruct_model_path "unsloth/Qwen2.5-Coder-1.5B-Instruct" \
        --finetuned_model_path "$HF_PATH" \
        --checkpoint_diff_output_path "$OUTPUT_PATH" \
        --is_peft False

    echo "[+] Finished generating Instruct model for Epoch $EPOCHS!"
    echo ""
done

echo "All epochs processed successfully!"