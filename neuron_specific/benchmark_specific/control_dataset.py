import os
from datasets import load_dataset
import pandas as pd
import json
import re
import random
from datasets import load_dataset
import warnings
from models.benchmark import Benchmark
from models.humaneval import HumanEval
from models.mceval_hard import MCEvalHard
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
    elif benchmark_name == "mceval_hard":
        return MCEvalHard(filepath="./benchmarks/mceval_hard.jsonl")
    else:
        raise ValueError(f"Invalid benchmark name: {benchmark_name}")

def get_target_dataset_jsonl(filepath="benchmarks/humaneval_plus.jsonl", benchmark_name="humaneval_plus"):
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

        elif benchmark_name == "mceval_hard":
            prompt_token_counts.append(len(enc.encode(data.get("prompt", ""), disallowed_special=())))
            solution_token_counts.append(len(enc.encode(data.get("canonical_solution", ""), disallowed_special=())))
            
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

                    elif benchmark_name == "mceval_hard":
                        candidate_prompt = sig_and_doc
                        candidate_solution = signature_only + "\n" + body_code
                        function_name = node.name
                        
                        cand_prompt_tokens = len(enc.encode(candidate_prompt, disallowed_special=()))
                        cand_sol_tokens = len(enc.encode(candidate_solution, disallowed_special=()))
                        
                        # STRICT RANGE FILTER
                        if not (min_prompt <= cand_prompt_tokens <= max_prompt): continue
                        if not (min_sol <= cand_sol_tokens <= max_sol): continue
                        if not candidate_solution.strip(): continue
                        
                        programming_languages = ["Shell", "Swift", "Tcl", "VimScript", "Visual Basic", "Scala", "CoffeeScript", "Common Lisp", "Dart", "Elixir", "Emacs Lisp", "Erlang", "F#", "Fortran", "Groovy", "Haskell", "Python", "R", "Racket", "Ruby", "Rust", "Java", "JavaScript", "C", "CPP", "Julia", "Kotlin", "C#", "PHP", "Lua", "Perl", "PowerShell"]
                        json_file = {
                            "id": f"{samples_collected}",
                            "task_id": f"{random.choice(programming_languages)}/{samples_collected}",
                            "prompt": candidate_prompt,
                            "canonical_solution": candidate_solution,
                            "tests": "def check(candidate):\n    pass",
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


def build_training_datasets():
    # 1. Stream from multiple languages in The Stack
    languages = [
        "data/python", 
        "data/java", 
        "data/javascript", 
        "data/c", 
        "data/c++", 
        "data/go",
        "data/rust",
        "data/c-sharp"
    ]

    lang_keywords = {
        "data/python": ["def ", "class "],
        "data/java": ["class ", "public ", "private "],
        "data/javascript": ["function ", "=>", "class "],
        "data/c": ["#include", "int main", "void "],
        "data/c++": ["#include", "class ", "void ", "std::"],
        "data/go": ["func ", "package "],
        "data/rust": ["fn ", "impl ", "pub "],
        "data/c-sharp": ["class ", "namespace ", "public "]
    }


    target_samples = 2000
    samples_per_lang = (target_samples // len(languages)) + 1 
    
    print(f"Downloading 10,000 background samples across {len(languages)} languages...")
    
    stack_texts = []
    for lang in languages:
        print(f"  -> Pulling from {lang}...")
        stack_dataset = load_dataset(
            "bigcode/the-stack", 
            data_dir=lang, 
            split="train", 
            streaming=True
        )
        # Smaller buffer size here since we stream multiple times
        stack_shuffled = stack_dataset.shuffle(seed=42, buffer_size=10000) 
        
        lang_count = 0
        valid_keywords = lang_keywords.get(lang, [])

        for row in stack_shuffled:
            if lang_count >= samples_per_lang:
                break
            content = row["content"]
            
            has_keyword = any(keyword in content for keyword in valid_keywords)
            if not has_keyword:
                continue
            
            if lang_count % 500 == 0:
                print(f"     ...Collected {lang_count}/{samples_per_lang} valid samples")

            stack_texts.append(content)
            lang_count += 1
            
        # If we hit our exact global target, break out entirely
        if len(stack_texts) >= 2000:
            break

    # Strictly enforce the 2000 limit just in case of rounding math
    stack_texts = stack_texts[:2000]
    print(f"Successfully collected {len(stack_texts)} diverse background samples!")

    # 3. Combine and shuffle everything together
    stack_df = pd.DataFrame({"content": stack_texts})
    
    # Shuffle so the model doesn't just memorize MCEval then The Stack sequentially
    stack_df = stack_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. Export to JSONL
    os.makedirs("./benchmarks/training", exist_ok=True)
    output_path = "./benchmarks/training/the-stack_training_data_2k_python_only.jsonl"
    stack_df.to_json(output_path, orient="records", lines=True)
    
    print(f"Success! Saved {len(stack_df)} instances to {output_path}")

    print("Extracting 227 instances from MCEval...")
    mceval_texts = []
    
    with open("./benchmarks/mceval_hard.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            full_json_string = json.dumps(data)
            mceval_texts.append(full_json_string)

    # 3. Combine and shuffle everything together
    combined_texts = mceval_texts + stack_texts
    combined_df = pd.DataFrame({"content": combined_texts})

    # 4. Export to JSONL
    os.makedirs("./benchmarks/training", exist_ok=True)
    output_path = "./benchmarks/training/mceval_and_2k_multi_language.jsonl"
    combined_df.to_json(output_path, orient="records", lines=True)
    
    print(f"Success! Saved {len(combined_df)} instances to {output_path}")


if __name__ == "__main__":
    # Example usage
    build_training_datasets()
    os._exit(0)