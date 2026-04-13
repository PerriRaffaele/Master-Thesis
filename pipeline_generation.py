import gc
import re
import torch
from neuron_specific.benchmark_specific.control_dataset import get_benchmark_by_name
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from neuron_specific.benchmark_specific.TSED import Calculate
import json


def parse_code_block(string: str) -> str:
    code_pattern = r"```[^\n]*\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)
    if match:
        return match.group(1)
    else:
        print("No code block found; returning full string")
        return string

def generate(messages: list[dict], model, tokenizer, max_new_tokens: int, temperature: float) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return parse_code_block(result)


def print_messages(messages: list[dict]):
    for message in messages:
        print(f"---------------------------- {message['role'].upper()} ----------------------------")
        print(message['content'])
        print("------------------------------------------------------------------------------------")


def export_jsonl(row, output_file):
    with open(output_file, 'a') as f:
        f.write(row.to_json() + '\n')

def masking_neurons(model, neurons_json_path):
    """
    Reads the top neurons JSON and physically zeroes out their weights in the model.
    """
    print(f"\n[+] Loading targeted neurons from {neurons_json_path}")
    
    with open(neurons_json_path, 'r') as f:
        neurons_dict = json.load(f)
        
    total_masked = 0
    
    # For down_proj, 'in_features' is the individual neuron inside the MLP.
    # By zeroing out a specific column, that neuron can no longer output any data.
    with torch.no_grad():
        for layer_name, neuron_entries in neurons_dict.items():
            if not neuron_entries:
                continue
            layer_idx = int(layer_name.split("_")[1])
            
            # FIX: extract just the integer indices from (idx, ap_score) tuples
            neuron_indices = [entry[0] for entry in neuron_entries]
            
            model.model.layers[layer_idx].mlp.down_proj.weight.data[:, neuron_indices] = 0.0
            total_masked += len(neuron_indices)
            
    print(f"[+] Successfully masked {total_masked} benchmark-specific neurons!\n")
    return model

def verify_masking(model, neurons_json_path):
    with open(neurons_json_path) as f:
        neurons_dict = json.load(f)
    
    with torch.no_grad():
        for layer_name, neuron_entries in neurons_dict.items():
            if not neuron_entries:
                continue
            layer_idx = int(layer_name.split("_")[1])
            neuron_indices = [entry[0] for entry in neuron_entries]
            
            col_norms = model.model.layers[layer_idx].mlp.down_proj.weight.data[:, neuron_indices].norm(dim=0)
            non_zero = (col_norms > 1e-6).sum().item()
            
            if non_zero > 0:
                print(f"[FAIL] {layer_name}: {non_zero}/{len(neuron_indices)} neurons NOT zeroed")
            else:
                print(f"[OK] {layer_name}: all {len(neuron_indices)} neurons zeroed")


if __name__ == '__main__':
    # Benchmark and Dataset
    benchmark_names = {
        1: "humaneval_plus",
        2: "mbpp_plus",
        3: "mceval_hard"
    }
    chosen_benchmark = 3
    benchmark_name = benchmark_names[chosen_benchmark]
    max_tokens = 1024
    temperature = 0.0
    iterations = 1
    benchmark = get_benchmark_by_name(benchmark_name)
    benchmark_df = benchmark.load_data()
    
    # Model
    model_ids = [
        # "./checkpoints_python_only_2k/Qwen2.5-Coder-1.5B-Instruct-Continuous_1",
        # "./checkpoints_python_only_2k/Qwen2.5-Coder-1.5B-Instruct-Continuous_2",
        # "./checkpoints_python_only_2k/Qwen2.5-Coder-1.5B-Instruct-Continuous_3",
        # "./checkpoints_python_only_2k/Qwen2.5-Coder-1.5B-Instruct-Continuous_4",
        # "./checkpoints_python_only_2k/Qwen2.5-Coder-1.5B-Instruct-Continuous_5",
        # "./checkpoints_python_only_2k/Qwen2.5-Coder-1.5B-Instruct-Continuous_6",
        # "./checkpoints_python_only_2k/Qwen2.5-Coder-1.5B-Instruct-Continuous_7",
        # "./checkpoints_python_only_2k/Qwen2.5-Coder-1.5B-Instruct-Continuous_8",
        "./checkpoints_python_only_2k/Qwen2.5-Coder-1.5B-Instruct-Continuous_9",
        "./checkpoints_python_only_2k/Qwen2.5-Coder-1.5B-Instruct-Continuous_10"
        ]
    
    for model_id in model_ids:
        print(f"\n===== Loading Model: {model_id} =====")
        if model_id.startswith("./checkpoints"):
            tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-1.5B-Instruct")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        thresholds = {
            # "0.22429510735316993": 5, # New training
            # "0.30368465675975936": 2,
            # "0.3395982194052997": 3,
            "0.37551178205084007": 4,
            # "0.4114253446963804": 5
        }

        for threshold, z in thresholds.items():
            mask_neurons = False
            if mask_neurons:
                print(f"\n\n==================== Running Pipeline with Threshold {threshold} ====================\n\n")
            else:
                print(f"\n\n==================== Running Baseline Pipeline WITHOUT Masking ====================\n\n")

            if 'model' in locals():
                del model
                gc.collect()              # Force Python to clear RAM
                torch.cuda.empty_cache()  # Force PyTorch to clear GPU VRAM

            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

            # check if model_id ends with a number
            neurons_file = None
            neurons_file = f"./results/benchmark_specific/{model_id.split('./')[1]}/benchmark_only/original_pure_memorization_neurons_TH{threshold}_Z{z}.json"
            # neurons_file = f"./results/benchmark_specific/{model_id.split('./')[1]}/new_dataset/mceval_hard_jsonl_top_benchmark_neurons_10000_{threshold}_Z{z}.json"
            if os.path.exists(neurons_file) and mask_neurons:
                model = masking_neurons(model, neurons_file)
                verify_masking(model, neurons_file)
            else:
                print(f"Warning: Could not find {neurons_file}. Running baseline evaluation without masking.")

            output_dir = './results/2k_new_training_python_only/'
            os.makedirs(output_dir, exist_ok=True)

            print(f"===== Arguments =====")
            print(f"Model: {model}")
            print(f"Benchmark: {benchmark_name}")
            print(f"Max tokens: {max_tokens}")
            print(f"Temperature: {temperature}")
            print(f"Output dir: {output_dir}")
            print(f"======================")

            passed, num_instances = 0, len(benchmark_df)
            #     # System prompt adapted from Reflexion (https://github.com/noahshinn/reflexion)
            system_prompt = f"""You are an AI that only responds with Python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation. You always return the signature and anything that came before it in the input prompt (such as the docstring, libraries, imports, and so on) along with the full implementation of the function. Write the output in a markdown code block. For example:\n```\n<your code here>\n```"""

            completion_kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "api_base": 'http://localhost:8000/v1',
            }

            for i in range(iterations):
                passed = 0
                print(f"Running iteration {i + 1}/{iterations}...")

                model_name_extracted = model_id.split("/")[-1].replace("-", "_")
                iteration_dir = os.path.join(output_dir, model_name_extracted, benchmark_name, f"iter_{i + 1}")
                os.makedirs(iteration_dir, exist_ok=True)

                for idx, row in benchmark_df.iterrows():
                    benchmark.row = row

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": benchmark.prompt()}
                    ]

                    solution = generate(
                        messages=messages,
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=max_tokens,
                        temperature=temperature
                    )

                    status, output = benchmark.run_tests(solution)

                    if status == True: passed += 1
                    print(
                        f"\n\n## Prompt {idx + 1}/{num_instances} - Current accuracy: {(passed / (idx + 1)) * 100:.2f}% ({passed}/{idx + 1})\n\n")
                    
                    if benchmark_name == "humaneval_plus" or benchmark_name == "mceval_hard":
                        canonical_full = row['prompt'] + row['canonical_solution']
                    else:
                        canonical_full = row["code"]

                    tsed_score = Calculate("python", solution, canonical_full, 1.0, 0.8, 1.0)

                    row['evaluated_prompt'] = benchmark.prompt()
                    row['evaluated_tests'] = benchmark.tests()
                    row['completion'] = solution
                    row['test_output'] = output
                    row['passed'] = status
                    row['tsed_score'] = tsed_score
                    if mask_neurons:
                        export_jsonl(row, os.path.join(iteration_dir, f"result_masked_original_pure_memorization_{threshold}_Z{z}.jsonl"))
                    else:
                        export_jsonl(row, os.path.join(iteration_dir, f"result_baseline_{benchmark_name}.jsonl"))