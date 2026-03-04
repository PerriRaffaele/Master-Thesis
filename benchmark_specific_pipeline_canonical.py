import os
import json
import torch
from models.mbpp import MBPP
from models.humaneval import HumanEval
from models.benchmark import Benchmark
from neuron_specific.benchmark_specific.background_dataset import build_background_dataset, decontaminate_background
from neuron_specific.benchmark_specific.compute_responses import compute_responses, get_mlp_hook
from neuron_specific.benchmark_specific.compute_expertise import compute_expertise
from neuron_specific.benchmark_specific.limit_expertise import limit_expertise
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

def get_benchmark_by_name(benchmark_name: str) -> Benchmark:
    """
    Get the benchmark object by its name
    Args:
        benchmark_name (str): Name of the benchmark
        benchmarks_dir (str): Directory where the benchmark files are stored
    Returns:
        Benchmark: The benchmark object
    """
    if benchmark_name == "humaneval_plus":
        return HumanEval()
    elif benchmark_name == "mbpp_plus":
        return MBPP()
    else:
        raise ValueError(f"Invalid benchmark name: {benchmark_name}")

def get_target_dataset_jsonl(filepath="benchmarks/humaneval_plus.jsonl"):
    """Loads the benchmark dataset directly as raw JSON strings."""
    print(f"Loading Target Dataset from JSONL: {filepath}...")
    target_texts = []
    
    if not os.path.exists(filepath):
        benchmark = get_benchmark_by_name(benchmark_name)
        benchmark_df = benchmark.load_data()
        benchmark_df.to_json(filepath, orient="records", lines=True)
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Append the raw JSON string
            target_texts.append(line.strip())
            
    print(f"Loaded {len(target_texts)} target benchmark samples as JSON strings.")
    return target_texts

# Global dictionary to store activations during the forward pass
activations_dict = defaultdict(list)

if __name__ == '__main__':
    # Benchmark and Dataset
    benchmark_names = {
        1: "humaneval_plus",
        2: "mbpp_plus"
    }
    chosen_benchmark = 2
    benchmark_name = benchmark_names[chosen_benchmark]
    benchmark_texts = get_target_dataset_jsonl(filepath=f"benchmarks/{benchmark_name}_dataset.jsonl")
    raw_background_dataset = build_background_dataset(num_samples=len(benchmark_texts), benchmark_name=benchmark_name)
    background_dataset = decontaminate_background(raw_background_dataset, benchmark_texts)
    
    # Model
    model_id = "unsloth/Qwen2.5-Coder-14B-Instruct"
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
    target_acts = compute_responses(model, tokenizer, activations_dict, benchmark_texts, desc="Target Passes")
    background_acts = compute_responses(model, tokenizer, activations_dict, background_dataset, desc="Background Passes")
    
    # Remove hooks to clean up memory
    for h in hooks: h.remove()
    
    # 5. Calculate AP and Extract Top Neurons
    ap_scores_per_layer = compute_expertise(target_acts, background_acts)
    top_benchmark_neurons = limit_expertise(ap_scores_per_layer, threshold=0.90)
    
    # 6. Save Results
    output_file = f"./results/benchmark_specific/{model_id}/{benchmark_name}_jsonl_top_benchmark_neurons.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(top_benchmark_neurons, f, indent=4)
        
    print(f"\nSuccess! Top benchmark neurons saved to {output_file}.")