import os
import json
import torch
from neuron_specific.benchmark_specific.compute_responses import compute_responses, get_mlp_hook
from neuron_specific.benchmark_specific.compute_expertise import compute_expertise
from neuron_specific.benchmark_specific.limit_expertise import limit_expertise
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from neuron_specific.benchmark_specific.control_dataset import build_control_dataset, get_target_dataset_jsonl, decontaminate_background
import random
from tqdm import tqdm
from analytics import analyze_and_plot_distribution

# Global dictionary to store activations during the forward pass
activations_dict = defaultdict(list)

if __name__ == '__main__':
    # Benchmark and Dataset
    benchmark_names = {
        1: "humaneval_plus",
        2: "mbpp_plus",
        3: "mceval_hard"
    }
    chosen_benchmark = 3
    benchmark_name = benchmark_names[chosen_benchmark]
    benchmark_texts = get_target_dataset_jsonl(filepath=f"benchmarks/{benchmark_name}_dataset.jsonl", benchmark_name=benchmark_name)
    # check if control dataset already exists in folder
    control_dataset_path = f"benchmarks/control_dataset/{benchmark_name}_control_dataset.jsonl"
    control_dataset = None
    if os.path.exists(control_dataset_path):
        with open(control_dataset_path, 'r', encoding='utf-8') as f:
            control_dataset = [line.strip() for line in f]
    else:
        control_dataset = build_control_dataset(benchmark_texts, num_samples=100000, benchmark_name=benchmark_name)
    
    # Get only 100000 samples for the control dataset to speed up the process
    sample_size = 10000
    random.seed(42)
    control_dataset = random.sample(control_dataset, sample_size)
    z_thresholds = [7, 8, 9, 10]
    # Model
    # model_id = "unsloth/Qwen2.5-Coder-14B-Instruct"
    for z_threshold in z_thresholds:
        model_id = "./checkpoints_no_lora/Qwen2.5-Coder-1.5B-Instruct-Continuous_10"
        if model_id.startswith("./checkpoints"):
            tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-1.5B-Instruct")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        print("===== Model Loaded =====")
        print(model)


        output_dir = './results/'
        os.makedirs(output_dir, exist_ok=True)

        print(f"===== Arguments =====")
        print(f"Model: {model}")
        print(f"Benchmark: {benchmark_name}")
        print(f"Output dir: {output_dir}")
        print(f"======================")

        # 3. Register Hooks to all MLP layers
        hooks = []
        print("Registering PyTorch hooks to MLP layers...")
        for i, layer in enumerate(model.model.layers):
            h = layer.mlp.down_proj.register_forward_pre_hook(get_mlp_hook(f"layer_{i}", activations_dict))
            hooks.append(h)
            
        # 4. Run Forward Passes
        batch_size = 128
        target_acts = compute_responses(model, tokenizer, activations_dict, benchmark_texts, desc="Target Passes", batch_size=batch_size)
        control_acts = compute_responses(model, tokenizer, activations_dict, control_dataset, desc="Control Passes", batch_size=batch_size)
        
        # Remove hooks to clean up memory
        for h in tqdm(hooks, desc="Removing Hooks"):
            h.remove()
        
        # 5. Calculate AP and Extract Top Neurons
        # threshold = 0.65
        ap_scores_per_layer = compute_expertise(target_acts, control_acts)
        # This will print the stats, save the graph, and return the exact Z=3 mathematical threshold
        derived_threshold = analyze_and_plot_distribution(ap_scores_per_layer, output_dir=output_dir, z_threshold=z_threshold)
        top_benchmark_neurons = limit_expertise(ap_scores_per_layer, threshold=derived_threshold)
        
        # 6. Save Results
        output_file = f"./results/benchmark_specific/{model_id}/new_dataset/{benchmark_name}_jsonl_top_benchmark_neurons_{sample_size}_{derived_threshold}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(top_benchmark_neurons, f, indent=4)
            
        print(f"\nSuccess! Top benchmark neurons saved to {output_file}.")