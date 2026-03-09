import os
import json
import torch
from neuron_specific.benchmark_specific.compute_responses import compute_responses, get_mlp_hook, compute_responses_chunks
from neuron_specific.benchmark_specific.compute_expertise import compute_expertise, compute_expertise_chunked
from neuron_specific.benchmark_specific.limit_expertise import limit_expertise
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from neuron_specific.benchmark_specific.control_dataset import build_control_dataset, get_target_dataset_jsonl, decontaminate_background
import random
from tqdm import tqdm

# Global dictionary to store activations during the forward pass
activations_dict = defaultdict(list)

if __name__ == '__main__':
    # Benchmark and Dataset
    benchmark_names = {
        1: "humaneval_plus",
        2: "mbpp_plus"
    }
    chosen_benchmark = 1
    benchmark_name = benchmark_names[chosen_benchmark]
    benchmark_texts = get_target_dataset_jsonl(filepath=f"benchmarks/{benchmark_name}_dataset.jsonl")
    # check if control dataset already exists in folder
    control_dataset_path = f"benchmarks/control_dataset/{benchmark_name}_control_dataset.jsonl"
    control_dataset = None
    if os.path.exists(control_dataset_path):
        with open(control_dataset_path, 'r', encoding='utf-8') as f:
            control_dataset = [line.strip() for line in f]
    else:
        raw_control_dataset = build_control_dataset(benchmark_texts, num_samples=1000000, benchmark_name=benchmark_name)
        control_dataset = decontaminate_background(raw_control_dataset, benchmark_texts)
    
    # Model
    # model_id = "unsloth/Qwen2.5-Coder-14B-Instruct"
    model_id = "unsloth/Qwen2.5-Coder-1.5B-Instruct"
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
    target_acts_paths = compute_responses_chunks(model, tokenizer, activations_dict, benchmark_texts, desc="Target Passes", batch_size=batch_size, type="Target")
    control_acts_paths = compute_responses_chunks(model, tokenizer, activations_dict, control_dataset, desc="Control Passes", batch_size=batch_size, type="Control")
    
    # Remove hooks to clean up memory
    for h in tqdm(hooks, desc="Removing Hooks"):
        h.remove()
    
    # 5. Calculate AP and Extract Top Neurons
    layer_names = [f"layer_{i}" for i in range(len(model.model.layers))]
    ap_scores_per_layer = compute_expertise_chunked(target_acts_paths, control_acts_paths, layer_names)
    top_benchmark_neurons = limit_expertise(ap_scores_per_layer, threshold=0.90)
    
    # 6. Save Results
    output_file = f"./results/benchmark_specific/{model_id}/new_dataset/{benchmark_name}_jsonl_top_benchmark_neurons.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(top_benchmark_neurons, f, indent=4)
        
    print(f"\nSuccess! Top benchmark neurons saved to {output_file}.")