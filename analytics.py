import json
import os

def compare_neuron_jsons(file_1: str, file_2: str, description: str):
    with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
        data_1 = json.load(f1)
        data_2 = json.load(f2)

    all_identical = True
    
    print(f"Comparing neurons => {description}")

    for layer in data_1.keys():
        if layer not in data_2:
            print(f"{layer} is missing in the generated JSON!")
            all_identical = False
            continue
            
        # Extract JUST the neuron IDs (index 0), ignoring the AP scores (index 1)
        neurons_1 = set(item[0] for item in data_1[layer])
        neurons_2 = set(item[0] for item in data_2[layer])
        
        # Compare the sets
        if neurons_1 == neurons_2:
            print(f"{layer}: EXACT MATCH ({len(neurons_1)} identical neurons)")
        else:
            all_identical = False
            # Calculate how many they actually share to give you some context
            shared = neurons_1.intersection(neurons_2)
            # Avoid division by zero if there are no neurons in the first set
            if len(neurons_1) == 0 and len(neurons_2) == 0:
                overlap_percentage = 100.0
            elif len(neurons_1) == 0:
                overlap_percentage = 0.0
            else:
                overlap_percentage = (len(shared) / len(neurons_1)) * 100
            print(f"{layer}: MISMATCH - Overlap: {overlap_percentage:.1f}% ({len(shared)}/{len(neurons_1)} shared)")

    print("\n--- Final Conclusion ---")
    if all_identical:
        print("Result: The exact same top neurons activated in both the Reading and Generating methods!\n")
    else:
        print("Result: The methods produced different sets of top neurons.\n")


def analyze_neuron_activation(filepath: str, total_neurons_per_layer: int, num_layers: int, model: str, dataset: str, specific: str):
    """
    Analyzes a single JSON file to provide presentation-ready statistics 
    about the activated neurons.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    total_network_neurons = total_neurons_per_layer * num_layers
    
    # Calculate total activated neurons and track layer density
    total_activated = 0
    layer_counts = {}
    
    for layer_name, neurons in data.items():
        count = len(neurons)
        layer_counts[layer_name] = count
        total_activated += count
        
    # Calculate percentage of total network
    percentage_of_network = (total_activated / total_network_neurons) * 100 if total_network_neurons > 0 else 0
    
    # Find the layer with the most activations (the "densest" layer)
    if layer_counts:
        densest_layer = max(layer_counts, key=layer_counts.get)
        max_activations = layer_counts[densest_layer]
    else:
        densest_layer = "N/A"
        max_activations = 0
        
    # Print the statistics cleanly for your presentation
    print(f"\n========================================================")
    print(f"STATISTICS FOR: {model} - {dataset} ({specific})")
    print(f"========================================================")
    print(f"Architecture: {num_layers} layers x {total_neurons_per_layer:,} neurons")
    print(f"Total possible MLP neurons: {total_network_neurons:,}\n")
    
    print(f"Total Experts Found: {total_activated:,}")
    print(f"Percentage of Network: {percentage_of_network:.4f}%")
    print(f"Densest Layer: {densest_layer} (with {max_activations} neurons)\n")
    
    print("Layer-by-Layer Breakdown:")
    
    # Sort layers numerically (e.g., layer_0, layer_1 ... layer_10) so it prints in order
    sorted_layers = sorted(data.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
    
    for layer in sorted_layers:
        count = layer_counts.get(layer, 0)
        if count > 0:
            # Adding a visual bar for quick scanning
            print(f"  {layer.ljust(10)}: {str(count).ljust(4)}")


if __name__ == '__main__':
    print("\n[ RUNNING COMPARISONS ]")
    compare_neuron_jsons(
        './results/benchmark_specific/unsloth/Qwen2.5-Coder-14B-Instruct/new_dataset/humaneval_plus_jsonl_top_benchmark_neurons_100000.json', 
        './results/benchmark_specific/unsloth/Qwen2.5-Coder-14B-Instruct/new_dataset/humaneval_plus_jsonl_top_benchmark_neurons_2000.json',
        "Humaneval Plus - Top Benchmark Neurons (100000 samples vs 2000 samples) - Qwen2.5-Coder-14B-Instruct"
        )
    compare_neuron_jsons(
        './results/benchmark_specific/unsloth/Qwen2.5-Coder-14B-Instruct/new_dataset/humaneval_plus_jsonl_top_benchmark_neurons_100000.json', 
        './results/benchmark_specific/unsloth/Qwen2.5-Coder-14B-Instruct/new_dataset/humaneval_plus_jsonl_top_benchmark_neurons_10000.json',
        "Humaneval Plus - Top Benchmark Neurons (100000 samples vs 2000 samples) - Qwen2.5-Coder-14B-Instruct"
        )
    analyze_neuron_activation(
        './results/benchmark_specific/unsloth/Qwen2.5-Coder-14B-Instruct/new_dataset/humaneval_plus_jsonl_top_benchmark_neurons_100000.json',
        total_neurons_per_layer=13824,
        num_layers=48,
        model="Qwen2.5-Coder-14B-Instruct",
        dataset="Humaneval Plus",
        specific="Top Benchmark Neurons (100k samples)"
    )
    analyze_neuron_activation(
        './results/benchmark_specific/unsloth/Qwen2.5-Coder-14B-Instruct/new_dataset/humaneval_plus_jsonl_top_benchmark_neurons_10000.json',
        total_neurons_per_layer=13824,
        num_layers=48,
        model="Qwen2.5-Coder-14B-Instruct",
        dataset="Humaneval Plus",
        specific="Top Benchmark Neurons (100k samples)"
    )
    analyze_neuron_activation(
        './results/benchmark_specific/unsloth/Qwen2.5-Coder-14B-Instruct/new_dataset/humaneval_plus_jsonl_top_benchmark_neurons_2000.json',
        total_neurons_per_layer=13824,
        num_layers=48,
        model="Qwen2.5-Coder-14B-Instruct",
        dataset="Humaneval Plus",
        specific="Top Benchmark Neurons (100k samples)"
    )
    # # CANONICAL VS COMPLETION (Humaneval Plus)
    # compare_neuron_jsons(
    #     './results/benchmark_specific/humaneval_plus_jsonl_completion_top_benchmark_neurons.json', 
    #     './results/benchmark_specific/humaneval_plus_jsonl_top_benchmark_neurons.json',
    #     "Completion vs Canonical (Humaneval Plus) - Qwen2.5-Coder-1.5B-Instruct"
    #     )
    # # CANONICAL VS COMPLETION (MBPP Plus)
    # compare_neuron_jsons(
    #     './results/benchmark_specific/mbpp_plus_jsonl_completion_top_benchmark_neurons.json', ''
    #     './results/benchmark_specific/mbpp_plus_jsonl_top_benchmark_neurons.json',
    #     "Completion vs Canonical (MBPP Plus) - Qwen2.5-Coder-1.5B-Instruct"
    #     )
    # # unsloth/Qwen2.5-Coder-14B-Instruct (MBPP Plus) VS unsloth/Qwen2.5-Coder-14B-Instruct (Python LAPE) 
    # compare_neuron_jsons(
    #     './results/benchmark_specific/unsloth/Qwen2.5-Coder-14B-Instruct/mbpp_plus_jsonl_top_benchmark_neurons.json', 
    #     './results/language_specific/unsloth/Qwen2.5-Coder-14B-Instruct/lape_python_neurons.json',
    #     "MBPP Plus vs Python LAPE - Qwen2.5-Coder-14B-Instruct"
    #     )
    # # codellama/CodeLlama-13b-Instruct-hf (MBPP Plus) VS codellama/CodeLlama-13b-Instruct-hf (Python LAPE) 
    # compare_neuron_jsons(
    #     './results/benchmark_specific/codellama/CodeLlama-13b-Instruct-hf/mbpp_plus_jsonl_top_benchmark_neurons.json', 
    #     './results/language_specific/codellama/CodeLlama-13b-Instruct-hf/lape_python_neurons.json',
    #     "MBPP Plus vs Python LAPE - CodeLlama-13b-Instruct-hf"
    #     )
    
    # print("\n[ ANALYZING INDIVIDUAL FILES ]")

    # qwen_2b_neurons_per_layer = 8960
    # qwen_2b_layers = 28
    # analyze_neuron_activation(
    #     './results/benchmark_specific/humaneval_plus_jsonl_top_benchmark_neurons.json', 
    #     total_neurons_per_layer=qwen_2b_neurons_per_layer, 
    #     num_layers=qwen_2b_layers,
    #     model="unsloth/Qwen2.5-Coder-1.5B-Instruct",
    #     dataset="Humaneval Plus",
    #     specific="Top Benchmark Neurons"
    # )
    
    # analyze_neuron_activation(
    #     './results/benchmark_specific/mbpp_plus_jsonl_top_benchmark_neurons.json', 
    #     total_neurons_per_layer=qwen_2b_neurons_per_layer, 
    #     num_layers=qwen_2b_layers,
    #     model="unsloth/Qwen2.5-Coder-1.5B-Instruct",
    #     dataset="MBPP Plus",
    #     specific="Top Benchmark Neurons"
    # )

    # qwen_14b_neurons_per_layer = 13824
    # qwen_14b_layers = 48
    # analyze_neuron_activation(
    #     './results/benchmark_specific/unsloth/Qwen2.5-Coder-14B-Instruct/mbpp_plus_jsonl_top_benchmark_neurons.json', 
    #     total_neurons_per_layer=qwen_14b_neurons_per_layer, 
    #     num_layers=qwen_14b_layers,
    #     model="Qwen2.5-Coder-14B-Instruct",
    #     dataset="MBPP Plus",
    #     specific="Top Benchmark Neurons"
    # )
    
    # analyze_neuron_activation(
    #     './results/language_specific/unsloth/Qwen2.5-Coder-14B-Instruct/lape_python_neurons.json', 
    #     total_neurons_per_layer=qwen_14b_neurons_per_layer, 
    #     num_layers=qwen_14b_layers,
    #     model="Qwen2.5-Coder-14B-Instruct",
    #     dataset="Multilingual LAPE",
    #     specific="Python Experts"
    # )

    # codellama_neurons_per_layer = 13824
    # codellama_layers = 40
    # analyze_neuron_activation(
    #     './results/benchmark_specific/codellama/CodeLlama-13b-Instruct-hf/mbpp_plus_jsonl_top_benchmark_neurons.json', 
    #     total_neurons_per_layer=codellama_neurons_per_layer, 
    #     num_layers=codellama_layers,
    #     model="codellama/CodeLlama-13b-Instruct-hf",
    #     dataset="MBPP Plus",
    #     specific="Top Benchmark Neurons"
    # )
    
    # analyze_neuron_activation(
    #     './results/language_specific/codellama/CodeLlama-13b-Instruct-hf/lape_python_neurons.json', 
    #     total_neurons_per_layer=codellama_neurons_per_layer, 
    #     num_layers=codellama_layers,
    #     model="codellama/CodeLlama-13b-Instruct-hf",
    #     dataset="Multilingual LAPE",
    #     specific="Python Experts"
    # )