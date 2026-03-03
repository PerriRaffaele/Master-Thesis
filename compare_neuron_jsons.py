import json

def compare_neuron_jsons(read_file: str, generate_file: str):
    with open(read_file, 'r') as f1, open(generate_file, 'r') as f2:
        read_data = json.load(f1)
        generate_data = json.load(f2)

    all_identical = True
    
    print(f"Comparing neurons...\n")

    for layer in read_data.keys():
        if layer not in generate_data:
            print(f"{layer} is missing in the generated JSON!")
            all_identical = False
            continue
            
        # Extract JUST the neuron IDs (index 0), ignoring the AP scores (index 1)
        read_neurons = set(item[0] for item in read_data[layer])
        generate_neurons = set(item[0] for item in generate_data[layer])
        
        # Compare the sets
        if read_neurons == generate_neurons:
            print(f"{layer}: EXACT MATCH ({len(read_neurons)} identical neurons)")
        else:
            all_identical = False
            # Calculate how many they actually share to give you some context
            shared = read_neurons.intersection(generate_neurons)
            overlap_percentage = (len(shared) / len(read_neurons)) * 100
            print(f"{layer}: MISMATCH - Overlap: {overlap_percentage:.1f}% ({len(shared)}/{len(read_neurons)} shared)")

    print("\n--- Final Conclusion ---")
    if all_identical:
        print("Result: The exact same top neurons activated in both the Reading and Generating methods!")
    else:
        print("Result: The methods produced different sets of top neurons.")

if __name__ == '__main__':
    # CANONICAL VS COMPLETION (Humaneval Plus)
    compare_neuron_jsons('./results/benchmark_specific/humaneval_plus_jsonl_completion_top_benchmark_neurons.json', './results/benchmark_specific/humaneval_plus_jsonl_top_benchmark_neurons.json')
    # CANONICAL VS COMPLETION (MBPP Plus)
    compare_neuron_jsons('./results/benchmark_specific/mbpp_plus_jsonl_completion_top_benchmark_neurons.json', './results/benchmark_specific/mbpp_plus_jsonl_top_benchmark_neurons.json')
    # unsloth/Qwen2.5-Coder-14B-Instruct (MBPP Plus) VS unsloth/Qwen2.5-Coder-14B-Instruct (Python LAPE) 
    compare_neuron_jsons('./results/benchmark_specific/unsloth/Qwen2.5-Coder-14B-Instruct/mbpp_plus_jsonl_top_benchmark_neurons.json', './results/language_specific/unsloth/Qwen2.5-Coder-14B-Instruct/lape_python_neurons.json')
    # codellama/CodeLlama-13b-Instruct-hf (MBPP Plus) VS codellama/CodeLlama-13b-Instruct-hf (Python LAPE) 
    compare_neuron_jsons('./results/benchmark_specific/codellama/CodeLlama-13b-Instruct-hf/mbpp_plus_jsonl_top_benchmark_neurons.json', './results/language_specific/codellama/CodeLlama-13b-Instruct-hf/lape_python_neurons.json')
    