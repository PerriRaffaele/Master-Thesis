import os
from datasets import load_dataset
import random
import json
import re
import random
from datasets import load_dataset
import warnings
from models.benchmark import Benchmark
from models.humaneval import HumanEval
from models.mbpp import MBPP
import ast
from tqdm import tqdm
import tiktoken

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


def build_control_dataset(benchmark_texts, num_samples=500, benchmark_name="humaneval_plus"):
    background_texts = []

    enc = tiktoken.get_encoding("cl100k_base")  # Accurate token counts
    
    # Calculate average lengths independently for prompt and solution
    print(f"Calculating average component lengths from {benchmark_name}...")
    prompt_token_counts = []
    solution_token_counts = []
    
    for text in benchmark_texts:
        data = json.loads(text)
        
        if benchmark_name == "humaneval_plus":
            # Count standard LLM tokens instantly
            prompt_token_counts.append(len(enc.encode(data.get("prompt", ""), disallowed_special=())))
            solution_token_counts.append(len(enc.encode(data.get("canonical_solution", ""), disallowed_special=())))
            
        elif benchmark_name == "mbpp_plus":
            prompt_token_counts.append(len(enc.encode(data.get("prompt", ""), disallowed_special=())))
            solution_token_counts.append(len(enc.encode(data.get("code", ""), disallowed_special=())))
            
    min_prompt = min(prompt_token_counts)
    max_prompt = max(prompt_token_counts)
    min_sol = min(solution_token_counts)
    max_sol = max(solution_token_counts)
        
    print(f"  -> Bounds Locked:")
    print(f"     Prompt Tokens: [{min_prompt} to {max_prompt}]")
    print(f"     Solution Tokens: [{min_sol} to {max_sol}]")
    
    # Stream from The Stack
    stack = load_dataset(
        "bigcode/the-stack", 
        data_dir="data/python", 
        split="train", 
        streaming=True,
    )
    stack = stack.shuffle(seed=42, buffer_size=10000)
    
    samples_collected = 0
    class_pattern = re.compile(r'\bclass\b')
    
    pbar = tqdm(total=num_samples, desc="Extracting Background Samples")
    
    try:
        for row in stack: 
            code = row["content"].strip()
            
            if class_pattern.search(code):
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)        
                try:
                    tree = ast.parse(code)
                except (SyntaxError, RecursionError, MemoryError):
                    # Skip files with invalid syntax, insane nesting, or massive memory footprints
                    continue
                
            for node in tree.body:
                if samples_collected >= num_samples:
                    break
                    
                if isinstance(node, ast.FunctionDef):
                    
                    func_source = ast.get_source_segment(code, node)
                    if not func_source: continue
                        
                    docstring = ast.get_docstring(node)
                    if not docstring: continue
                    
                    if len(node.body) > 1:
                        body_start_idx = node.body[1].lineno - node.lineno
                    else:
                        continue 
                        
                    lines = func_source.split('\n')
                    sig_and_doc = '\n'.join(lines[:body_start_idx])
                    body_code = '\n'.join(lines[body_start_idx:])
                    sig_end_idx = node.body[0].lineno - node.lineno
                    signature_only = '\n'.join(lines[:sig_end_idx])
                    
                    # 4. Filter and Format based on Benchmark
                    if benchmark_name == "humaneval_plus":
                        candidate_prompt = sig_and_doc
                        function_name = node.name
                        candidate_solution = body_code
                        
                        # Use tiktoken to check the size
                        cand_prompt_tokens = len(enc.encode(candidate_prompt, disallowed_special=()))
                        cand_sol_tokens = len(enc.encode(candidate_solution, disallowed_special=()))
                        
                        # STRICT RANGE FILTER
                        if not (min_prompt <= cand_prompt_tokens <= max_prompt): continue
                        if not (min_sol <= cand_sol_tokens <= max_sol): continue
                        if not candidate_solution.strip(): continue
                        
                        json_file = {
                            "task_id": f"Background_Task/{samples_collected}",
                            "prompt": candidate_prompt,
                            "canonical_solution": candidate_solution,
                            "test": "def check(candidate):\n    pass",
                            "entry_point": f"{function_name}"
                        }
                        
                    elif benchmark_name == "mbpp_plus":
                        candidate_prompt = docstring if docstring else ""
                        candidate_solution = signature_only + "\n" + body_code
                        function_name = node.name
                        
                        cand_prompt_tokens = len(enc.encode(candidate_prompt, disallowed_special=()))
                        cand_sol_tokens = len(enc.encode(candidate_solution, disallowed_special=()))
                        
                        # STRICT RANGE FILTER
                        if not (min_prompt <= cand_prompt_tokens <= max_prompt): continue
                        if not (min_sol <= cand_sol_tokens <= max_sol): continue
                        if not candidate_solution.strip(): continue
                        
                        json_file = {
                            "task_id": f"Background_Task/{samples_collected}",
                            "code": candidate_solution,
                            "prompt": candidate_prompt,
                            "source_file": f"{function_name}.py",
                            "test_imports": [],
                            "test_list": ["assert True"],
                            "test": "assertion(function(*), exp, 0):\n    pass"
                        }

                    background_texts.append(json.dumps(json_file))
                    samples_collected += 1

                    pbar.update(1)

            if samples_collected >= num_samples:
                break
    finally:
        pbar.close()
        del stack

    print(f"Built control dataset with {len(background_texts)} {benchmark_name}-formatted samples!")
    # save as JSONL for future use
    output_path = f"benchmarks/control_dataset/{benchmark_name}_control_dataset.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for text in background_texts:
            f.write(text + "\n")
    return background_texts


def decontaminate_background(background_texts, benchmark_json_strings):
    """
    Removes any background text that contains the function signatures 
    from the benchmark dataset.
    """
    print("Decontaminating background dataset using function signatures...")
    clean_background = []
    
    # 1. Extract the unique function signatures from the benchmark
    signature_pattern = re.compile(r"(def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)(?:\s*->\s*[^:]+)?\s*:)")
    benchmark_signatures = []
    
    for json_str in benchmark_json_strings:
        try:
            data = json.loads(json_str)
            prompt = data.get("prompt", "")
            
            # Search the prompt for the function signature
            match = signature_pattern.search(prompt)
            if match:
                # Clean up any weird spacing to make matching more robust
                signature = match.group(1).strip()
                benchmark_signatures.append(signature)
        except json.JSONDecodeError:
            continue

    print(f"Extracted {len(benchmark_signatures)} unique function signatures to filter against.")
    print("Example function signature extracted from benchmark:")
    if len(benchmark_signatures) > 0:
        print(benchmark_signatures[0])
    # 2. Filter the background dataset
    for bg_text in background_texts:
        is_contaminated = False
        
        for signature in benchmark_signatures:
            # If the specific benchmark signature appears in the background code, flag it
            if signature in bg_text:
                is_contaminated = True
                break
        
        if not is_contaminated:
            clean_background.append(bg_text)
            
    removed_count = len(background_texts) - len(clean_background)
    print(f"Decontamination complete. Removed {removed_count} overlapping/leaked samples.")

    return clean_background

if __name__ == "__main__":
    # Example usage
    benchmark_name = "humaneval_plus"
    benchmark_texts = get_target_dataset_jsonl(filepath=f"benchmarks/{benchmark_name}_dataset.jsonl")
    build_control_dataset(benchmark_texts, num_samples=1000000, benchmark_name=benchmark_name)
    
    # Force exit to kill any lingering HuggingFace streaming background threads
    os._exit(0)