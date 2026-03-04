import os
from datasets import load_dataset
import random
import json
import re
import random
from datasets import load_dataset
import warnings
from benchmark_specific_pipeline_canonical import get_target_dataset_jsonl
import ast
from tqdm import tqdm

def build_background_dataset_old(num_samples=500, benchmark_name="humaneval_plus"):
    background_texts = []
    
    # Calculate an even 3-way split
    english_needed = num_samples // 3
    raw_python_needed = num_samples // 3
    json_python_needed = num_samples - english_needed - raw_python_needed
    
    # 1. Fetch Plain English (WikiText)
    print(f"Downloading {english_needed} English background samples...")
    wiki = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    
    english_samples = 0
    for row in wiki:
        text = row["text"].strip()
        # Only take substantial paragraphs
        if len(text) > 200: 
            background_texts.append(text[:1000]) # Cap length to match benchmark sizes
            english_samples += 1
        if english_samples >= english_needed:
            break

    # 2. Fetch Boilerplate/Non-Algorithmic Python (The Stack Smol)
    print(f"Downloading {raw_python_needed + json_python_needed} Python background samples...")
    stack = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
    
    raw_python_samples = 0
    json_python_samples = 0
    # Keywords indicating boilerplate/web/config code, NOT algorithms
    boring_keywords = ["django", "flask", "setup(", "argparse", "sqlalchemy", "route(", "html"]
    
    for row in stack:
        code = row["content"]
        if any(kw in code.lower() for kw in boring_keywords) and len(code) > 100:
            truncated_code = code[:1000] 
            
            # Split into Raw Python and JSON-formatted Python
            if raw_python_samples < raw_python_needed:
                background_texts.append(truncated_code)
                raw_python_samples += 1
                
            elif json_python_samples < json_python_needed:
                # Wrap the boilerplate code in a dummy JSON structure that mimics HumanEval
                # We split the code in half to populate a fake "prompt" and "solution"
                split_point = len(truncated_code) // 2
                if benchmark_name == "humaneval_plus":
                    dummy_json = {
                        "task_id": f"Background_Task/{json_python_samples}",
                        "prompt": truncated_code[:split_point],
                        "canonical_solution": truncated_code[split_point:],
                        "test": "def check(candidate): pass",
                        "entry_point": "dummy_function"
                    }
                elif benchmark_name == "mbpp_plus":
                    dummy_json = {
                        "task_id": f"Background_Task/{json_python_samples}",
                        "code": truncated_code[split_point:],
                        "prompt": truncated_code[:split_point],
                        "source_file": "dummy.py",
                        "test_imports": "import dummy",
                        "test_list": ["def test(): pass"],
                        "test": "def check(candidate): pass",
                    }
                # Convert dict to JSON string and append
                background_texts.append(json.dumps(dummy_json))
                json_python_samples += 1
                
            if raw_python_samples >= raw_python_needed and json_python_samples >= json_python_needed:
                break

    # Shuffle the dataset so English, Raw Python, and JSON Python are mixed randomly
    random.shuffle(background_texts)
    print(f"Built background dataset with {len(background_texts)} samples!")
    
    return background_texts


def build_background_dataset(benchmark_texts, num_samples=500, benchmark_name="humaneval_plus"):
    background_texts = []
    
    # Calculate average lengths independently for prompt and solution
    print(f"Calculating average component lengths from {benchmark_name}...")
    total_prompt_len = 0
    total_solution_len = 0
    
    for text in benchmark_texts:
        data = json.loads(text)
        
        if benchmark_name == "humaneval_plus":
            total_prompt_len += len(data.get("prompt", ""))
            total_solution_len += len(data.get("canonical_solution", ""))
        elif benchmark_name == "mbpp_plus":
            total_prompt_len += len(data.get("prompt", ""))
            total_solution_len += len(data.get("code", ""))
            
    num_bench_samples = max(1, len(benchmark_texts)) # Prevent division by zero
    avg_prompt_len = int(total_prompt_len / num_bench_samples)
    avg_solution_len = int(total_solution_len / num_bench_samples)
        
    print(f"  -> Averages Locked: Prompt = {avg_prompt_len} chars, Solution = {avg_solution_len} chars")
    print(f"Downloading {num_samples} diverse Python background samples from The Stack...")
    
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
                    if not func_source:
                        continue
                        
                    docstring = ast.get_docstring(node)
                    
                    if docstring and len(node.body) > 1:
                        body_start_idx = node.body[1].lineno - node.lineno
                    elif not docstring and len(node.body) > 0:
                        body_start_idx = node.body[0].lineno - node.lineno
                    else:
                        continue 
                        
                    lines = func_source.split('\n')
                    
                    sig_and_doc = '\n'.join(lines[:body_start_idx])
                    body_code = '\n'.join(lines[body_start_idx:])
                    
                    if docstring:
                        sig_end_idx = node.body[0].lineno - node.lineno
                        signature_only = '\n'.join(lines[:sig_end_idx])
                    else:
                        signature_only = sig_and_doc
                    
                    # 4. Filter and Format based on Benchmark
                    if benchmark_name == "humaneval_plus":
                        candidate_prompt = sig_and_doc
                        function_name = node.name
                        candidate_solution = body_code
                        
                        # STRICT FILTER: Discard if it exceeds the benchmark averages
                        if len(candidate_prompt) > avg_prompt_len or len(candidate_solution) > avg_solution_len:
                            continue
                            
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
                        
                        # STRICT FILTER: Discard if it exceeds the benchmark averages
                        if len(candidate_prompt) > avg_prompt_len or len(candidate_solution) > avg_solution_len:
                            continue
                            
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
        # Explicitly delete the streaming iterator so HuggingFace background
        # threads (prefetch workers, file handles) are released before we return.
        pbar.close()
        del stack

    print(f"Built background dataset with {len(background_texts)} {benchmark_name}-formatted samples!")
    # save as JSONL for future use
    output_path = f"benchmarks/control_dataset/{benchmark_name}_background_dataset.jsonl"
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
    build_background_dataset(benchmark_texts, num_samples=1000000, benchmark_name=benchmark_name)
    
    # Force exit to kill any lingering HuggingFace streaming background threads
    os._exit(0)