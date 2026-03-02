import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
from models.humaneval import HumanEval
from neuron_specific.python_specific.compute_responses import compute_language_responses
from neuron_specific.python_specific.compute_expertise import compute_lape_scores
from neuron_specific.python_specific.limit_expertise import limit_python_expertise
from models.mbpp import MBPP
from models.humaneval import HumanEval
from models.benchmark import Benchmark

def get_benchmark_by_name(benchmark_name: str) -> Benchmark:
    """
    Get the benchmark object by its name
    Args:
        benchmark_name (str): Name of the benchmark
        benchmarks_dir (str): Directory where the benchmark files are stored
    Returns:
        Benchmark: The benchmark object
    """
    if benchmark_name == "humaneval":
        return HumanEval()
    elif benchmark_name == "mbpp":
        return MBPP()
    else:
        raise ValueError(f"Invalid benchmark name: {benchmark_name}")
    
def get_multilingual_dataset(benchmark_name, filepath, num_samples=150):
    """
    Loads parallel algorithmic problems in multiple languages.
    We use MultiPL-E, which contains HumanEval translated into many languages.
    """
    multilingual_texts = {}
    print(f"  Fetching python from local file: {filepath}...")
    py_texts = []
    
    if not os.path.exists(filepath):
        benchmark = get_benchmark_by_name(benchmark_name)
        benchmark_df = benchmark.load_data()
        benchmark_df.to_json(filepath, orient="records", lines=True)
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                # Extract the pure Python code
                prompt = data.get("prompt", "")
                full_code = prompt 
                
                py_texts.append(full_code)
                if len(py_texts) >= num_samples:
                    break
            except json.JSONDecodeError:
                continue
                
    multilingual_texts["python"] = py_texts
    print(f"    Loaded {len(py_texts)} Python samples.")
    print("Loading multilingual parallel datasets (MultiPL-E)...")
    
    # MultiPL-E split names mapping
    other_languages = {
        "java": f"{benchmark_name}-java",
        "cpp": f"{benchmark_name}-cpp",
        "javascript": f"{benchmark_name}-js",
        "ruby": f"{benchmark_name}-rb",
        "go": f"{benchmark_name}-go",
        "php": f"{benchmark_name}-php"
    }
    
    for lang, split_name in other_languages.items():
        print(f"  Fetching {lang} (MultiPL-E: {split_name})...")
        try:
            ds = load_dataset("nuprl/MultiPL-E", split_name, split="test")
            texts = []
            
            for row in ds:
                # MultiPL-E stores the raw translated code in the "prompt" key
                texts.append(row["prompt"])
                if len(texts) >= num_samples:
                    break
                    
            multilingual_texts[lang] = texts
            print(f"    Loaded {len(texts)} {lang.capitalize()} samples.")
            
        except Exception as e:
            print(f"    Failed to load {lang}: {e}")
            
    return multilingual_texts


# Dictionaries to store token counts per language
active_token_counts = defaultdict(lambda: defaultdict(int))
total_token_counts = defaultdict(lambda: defaultdict(int))

if __name__ == '__main__':
    benchmark_names = {
        1: "humaneval",
        2: "mbpp"
    }
    benchmark = benchmark_names[1] 
    # Fetch Parallel Multilingual Dataset
    multilingual_texts = get_multilingual_dataset(benchmark, filepath=f"benchmarks/{benchmark}_plus_dataset.jsonl", num_samples=150)
    
    # Load Model
    model_id = 'unsloth/Qwen2.5-Coder-1.5B-Instruct'
    print(f"\nLoading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    
    # Run Forward Passes & Capture Token Activations
    compute_language_responses(model, tokenizer, multilingual_texts, active_token_counts, total_token_counts)
    
    # Calculate LAPE & Filter for Python Experts
    layers = [f"layer_{i}" for i in range(len(model.model.layers))]
    lape_scores_per_layer = compute_lape_scores(layers, active_token_counts, total_token_counts)
    top_python_neurons = limit_python_expertise(lape_scores_per_layer, max_entropy=0.15)
    
    # Save Results
    output_file = "./results/language_specific/lape_python_neurons.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(top_python_neurons, f, indent=4)
        
    print(f"\nSuccess! Python-specific neurons saved to {output_file}.")