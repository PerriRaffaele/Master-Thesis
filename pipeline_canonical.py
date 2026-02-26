import os
import re
import json
import torch
from typing import Tuple
from litellm import completion
from models.mbpp import MBPP
from models.humaneval import HumanEval
from models.benchmark import Benchmark
from neuron_specific.benchmark_specific.background_dataset import build_background_dataset
from neuron_specific.benchmark_specific.compute_responses import compute_responses, get_mlp_hook
from neuron_specific.benchmark_specific.compute_expertise import compute_expertise
from neuron_specific.benchmark_specific.limit_expertise import limit_expertise
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

def get_target_dataset(benchmark_name="humaneval"):
    """Loads the benchmark prompts and canonical solutions."""
    print(f"Loading Target Dataset: {benchmark_name}...")
    if benchmark_name == "humaneval":
        benchmark = HumanEval()
    elif benchmark_name == "mbpp":
        benchmark = MBPP()
    else:
        raise ValueError(f"Invalid benchmark name: {benchmark_name}")
        
    df = benchmark.load_data()
    target_texts = []
    
    for _, row in df.iterrows():
        benchmark._row = row # Set the row to use the class methods
        # Combine prompt and solution to get the full algorithmic context
        full_text = benchmark.prompt() + str(row.get('canonical_solution', ''))
        target_texts.append(full_text)
        
    print(f"Loaded {len(target_texts)} target benchmark samples.")
    return target_texts

# Global dictionary to store activations during the forward pass
activations_dict = defaultdict(list)

if __name__ == '__main__':
    # Benchmark and Dataset
    benchmark_names = {
        1: "humaneval",
        2: "mbpp"
    }
    chosen_benchmark = 1
    benchmark_name = benchmark_names[chosen_benchmark]
    benchmark_texts = get_target_dataset(benchmark_name)
    backgound_dataset = build_background_dataset(num_samples=len(benchmark_texts))
    
    # Model
    # model = 'hosted_vllm/unsloth/Qwen2.5-Coder-1.5B-Instruct'
    model_id = 'unsloth/Qwen2.5-Coder-1.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")


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
    target_acts = compute_responses(model, tokenizer, activations_dict, benchmark_texts, desc="Target Passes")
    background_acts = compute_responses(model, tokenizer, activations_dict, backgound_dataset, desc="Background Passes")
    
    # Remove hooks to clean up memory
    for h in hooks: h.remove()
    
    # 5. Calculate AP and Extract Top Neurons
    ap_scores_per_layer = compute_expertise(target_acts, background_acts)
    top_benchmark_neurons = limit_expertise(ap_scores_per_layer, top_k=50)
    
    # 6. Save Results
    output_file = "./results/benchmark_specific/canonical_top_benchmark_neurons.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(top_benchmark_neurons, f, indent=4)
        
    print(f"\nSuccess! Top benchmark neurons saved to {output_file}.")