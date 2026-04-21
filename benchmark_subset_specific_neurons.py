from analytics import analyze_pass_distribution, split_benchmark_by_memorization, analyze_and_plot_distribution
import os
import json
import torch
from neuron_specific.benchmark_specific.compute_responses import compute_responses, get_mlp_hook
from neuron_specific.benchmark_specific.compute_expertise import compute_expertise
from neuron_specific.benchmark_specific.limit_expertise import limit_expertise
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm


activations_dict = defaultdict(list)

if __name__ == '__main__':
    # 1. Define Paths
    benchmark_name = "mceval_hard"
    all_training_path = "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl"
    pl_only_path = "./results/2k_new_training_multi_language/Qwen2.5_Coder_1.5B_Instruct_Continuous_2/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl"
    raw_benchmark_path = f"benchmarks/{benchmark_name}.jsonl" 
    
    # 2. Extract exactly which tasks were memorized
    memorized_ids, _, _ = analyze_pass_distribution(all_training_path, pl_only_path, benchmark_name, "Qwen2.5-Coder-1.5B-Instruct_Continuous_3")
    
    if not memorized_ids:
        print("[!] No memorized tasks found. Exiting.")
        exit()
        
    print(f"\n[+] Splitting {benchmark_name} into {len(memorized_ids)} Target tasks and Control tasks...")
    
    # Target = Memorized tasks | Control = Non-memorized MCEval tasks
    target_texts, control_texts = split_benchmark_by_memorization(raw_benchmark_path, memorized_ids)
    
    print(f"    -> Target (Memorized) Dataset Size: {len(target_texts)}")
    print(f"    -> Control (Non-Memorized) Dataset Size: {len(control_texts)}")

    z_thresholds = [6,7,8] # Adjusted to more standard Z-scores based on your previous logs
    for z_threshold in z_thresholds:
        model_id = "./checkpoints_with_2k_multi/Qwen2.5-Coder-1.5B-Instruct-Continuous_3"
        
        # Load Model (Only load it ONCE outside the loop to save massive amounts of time!)
        print(f"\n===== Loading Model: {model_id} =====")
        if model_id.startswith("./checkpoints"):
            tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-1.5B-Instruct")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        
        output_dir = './results/'
        os.makedirs(output_dir, exist_ok=True)

        # 4. Register Hooks
        hooks = []
        print("\nRegistering PyTorch hooks to MLP layers...")
        for i, layer in enumerate(model.model.layers):
            h = layer.mlp.down_proj.register_forward_pre_hook(get_mlp_hook(f"layer_{i}", activations_dict))
            hooks.append(h)
            
        # 5. Run Forward Passes
        batch_size = 128
        
        # Pass our newly split datasets!
        target_acts = compute_responses(model, tokenizer, activations_dict, target_texts, desc="Target (Memorized) Passes", batch_size=batch_size)
        control_acts = compute_responses(model, tokenizer, activations_dict, control_texts, desc="Control (Non-Memorized) Passes", batch_size=batch_size)
        
        # Remove hooks
        for h in tqdm(hooks, desc="Removing Hooks"):
            h.remove()
        
        # 6. Calculate Expertise
        # Only neurons strictly reacting to the *memorized* nature of the target text will score high!
        ap_scores_per_layer = compute_expertise(target_acts, control_acts)

        # 7. Extract for different Z-Thresholds
        print(f"\n--- Processing Z-Threshold: {z_threshold} ---")
        
        # Analyze and extract based on this specific Z
        derived_threshold = analyze_and_plot_distribution(ap_scores_per_layer, output_dir=output_dir, z_threshold=z_threshold)
        top_benchmark_neurons = limit_expertise(ap_scores_per_layer, threshold=derived_threshold)
        
        # Save Results
        output_file = f"./results/benchmark_specific/{model_id}/benchmark_only/original_pure_memorization_neurons_TH{derived_threshold}_Z{z_threshold}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(top_benchmark_neurons, f, indent=4)
            
        print(f"Success! Pure memorization neurons saved to {output_file}.")