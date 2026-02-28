from datasets import load_dataset
import random
import json
import re
import json
import random
from datasets import load_dataset

def build_background_dataset(num_samples=500, benchmark_name="humaneval_plus"):
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