import json
import os
import numpy as np
import matplotlib.pyplot as plt

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

def calculate_metrics(file_path):
    """Reads a result.jsonl file and calculates Accuracy and Mean TSED."""
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return None, None, 0

    passed_count = 0
    total_count = 0
    tsed_scores = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            
            data = json.loads(line)
            total_count += 1
            
            if data.get('passed', False):
                passed_count += 1
            
            tsed = data.get('tsed_score')
            if tsed is not None:
                tsed_scores.append(tsed)
                
    accuracy = (passed_count / total_count) * 100 if total_count > 0 else 0.0
    mean_tsed = sum(tsed_scores) / len(tsed_scores) if tsed_scores else 0.0
    
    return accuracy, mean_tsed, total_count

def run_comparison_masked(baseline_path, other_path, description="MASKED"):
    print("======================================================")
    print(f"MECHANISTIC INTERPRETABILITY: {description} REPORT")
    print("======================================================\n")
    
    # 1. Calculate Baseline Metrics
    base_acc, base_tsed, base_total = calculate_metrics(baseline_path)
    if base_total == 0: return
    
    # 2. Calculate Other Metrics
    other_acc, other_tsed, other_total = calculate_metrics(other_path)
    if other_total == 0: return

    # 3. Calculate Differences
    acc_diff = other_acc - base_acc
    tsed_diff = other_tsed - base_tsed

    # 4. Print the Thesis-Ready Report
    print(f"Dataset Size: {base_total} prompts evaluated.\n")
    
    print("ACCURACY (Pass Rate %)")
    print(f"  Baseline Model:   {base_acc:.2f}%")
    print(f"  Masked:      {other_acc:.2f}%")
    print(f"  -------------------------")
    print(f"  Absolute Impact:  {acc_diff:+.2f}%\n")

    print("TSED SIMILARITY (Mean Score)")
    print(f"  Baseline Model:   {base_tsed:.4f}")
    print(f"  Masked:      {other_tsed:.4f}")
    print(f"  -------------------------")
    print(f"  Absolute Impact:  {tsed_diff:+.4f}\n")

def run_comparison_models(baseline_path, other_path, description="MASKED"):
    print("======================================================")
    print(f"MECHANISTIC INTERPRETABILITY: {description} REPORT")
    print("======================================================\n")
    
    # 1. Calculate Baseline Metrics
    base_acc, base_tsed, base_total = calculate_metrics(baseline_path)
    if base_total == 0: return
    
    # 2. Calculate Other Metrics
    other_acc, other_tsed, other_total = calculate_metrics(other_path)
    if other_total == 0: return

    # 3. Calculate Differences
    acc_diff = other_acc - base_acc
    tsed_diff = other_tsed - base_tsed

    # 4. Print the Thesis-Ready Report
    print(f"Dataset Size: {base_total} prompts evaluated.\n")
    
    print("ACCURACY (Pass Rate %)")
    print(f"  Model 1:   {base_acc:.2f}%")
    print(f"  Model 2:      {other_acc:.2f}%")
    print(f"  -------------------------")
    print(f"  Absolute Impact:  {acc_diff:+.2f}%\n")

    print("TSED SIMILARITY (Mean Score)")
    print(f"  Model 1:   {base_tsed:.4f}")
    print(f"  Model 2:      {other_tsed:.4f}")
    print(f"  -------------------------")
    print(f"  Absolute Impact:  {tsed_diff:+.4f}\n")

def run_comparison_more_models(models_dict: dict, description="MULTIPLE MODELS"):
    print("=========================================================================")
    print(f"MECHANISTIC INTERPRETABILITY: {description} REPORT")
    print("=========================================================================\n")
    
    # 1. Gather all metrics
    results = {}
    for model_name, path in models_dict.items():
        acc, tsed, total = calculate_metrics(path)
        if total > 0:
            results[model_name] = {'acc': acc, 'tsed': tsed, 'total': total}
        else:
            print(f"[!] Warning: No valid data found for '{model_name}' at {path}\n")

    if len(results) < 2:
        print("Not enough valid models to compare. Need at least 2.")
        return

    model_names = list(results.keys())

    # 2. Print Absolute Metrics Table
    print("--- 1. ABSOLUTE METRICS ---")
    header_str = f"{'Model Name':<45} | {'Accuracy (%)':<15} | {'TSED Score':<15} | {'Samples'}"
    print(header_str)
    print("-" * len(header_str))
    for name in model_names:
        metrics = results[name]
        print(f"{name:<45} | {metrics['acc']:>14.2f}% | {metrics['tsed']:>15.4f} | {metrics['total']}")
    print("\n")

    # 3. Print Accuracy Comparison Matrix
    print("--- 2. ACCURACY IMPACT (Base vs Comparison) ---")
    print("How to read: (Column Model Accuracy) - (Row Model Accuracy)")
    
    col_headers = " | ".join([f"{name[:15]:>15}" for name in model_names]) # Truncate to 15 chars for neatness
    header_str = f"{'Base Model (Row) \\ Compare (Col)':<35} | " + col_headers
    print(header_str)
    print("-" * len(header_str))
    
    for row_name in model_names:
        row_str = f"{row_name[:35]:<35} | "
        for col_name in model_names:
            if row_name == col_name:
                row_str += f"{'-':>15} | "
            else:
                acc_diff = results[col_name]['acc'] - results[row_name]['acc']
                row_str += f"{acc_diff:>+14.2f}% | "
        print(row_str)
    print("\n")

    # 4. Print TSED Comparison Matrix
    print("--- 3. TSED SIMILARITY IMPACT (Base vs Comparison) ---")
    print("How to read: (Column Model TSED) - (Row Model TSED)")
    
    header_str = f"{'Base Model (Row) \\ Compare (Col)':<35} | " + col_headers
    print(header_str)
    print("-" * len(header_str))
    
    for row_name in model_names:
        row_str = f"{row_name[:35]:<35} | "
        for col_name in model_names:
            if row_name == col_name:
                row_str += f"{'-':>15} | "
            else:
                tsed_diff = results[col_name]['tsed'] - results[row_name]['tsed']
                row_str += f"{tsed_diff:>+15.4f} | "
        print(row_str)
    print("\n")

def count_detected_neurons(filepath: str):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return 0
        
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    total_count = sum(len(neurons) for neurons in data.values())
    return total_count

def analyze_and_plot_distribution(ap_scores_per_layer, output_dir="./results/", z_threshold=3):
    print("\n======================================================")
    print("NEURON EXPERTISE STATISTICAL ANALYSIS")
    print("======================================================")
    
    # 1. Flatten all 250,880 scores into a single 1D array
    all_scores = np.concatenate([np.array(scores) for layer, scores in ap_scores_per_layer.items()])
    
    # 2. Calculate the global Mean and Standard Deviation
    mu = np.mean(all_scores)
    sigma = np.std(all_scores)
    
    print(f"Total Neurons Analyzed: {len(all_scores)}")
    print(f"Global Mean (μ): {mu:.6f}")
    print(f"Standard Deviation (σ): {sigma:.6f}\n")
    
    # 3. Calculate how many neurons fall into strict Z-score outlier buckets
    print("--- Mathematically Derived Thresholds ---")
    z_scores_to_check = [2, 3, 4, 5]
    
    for z in z_scores_to_check:
        threshold = mu + (z * sigma)
        num_outliers = np.sum(all_scores > threshold)
        percentage = (num_outliers / len(all_scores)) * 100
        print(f"Z-Score >= {z} | Threshold = {threshold:.4f} | Masked Neurons: {num_outliers} ({percentage:.2f}%)")
        
    # 4. Generate the Histogram (Using Log Scale for the Y-axis)
    plt.figure(figsize=(12, 7))
    
    # We use a log scale because neural expertise follows a heavy-tailed distribution.
    # Most neurons will cluster near 0, and a tiny few will stretch far to the right.
    plt.hist(all_scores, bins=100, log=True, color='#4A90E2', edgecolor='black', alpha=0.7)
    
    # Draw vertical lines for the Mean and the Z=3 outlier threshold
    plt.axvline(mu, color='red', linestyle='dashed', linewidth=2, label=f'Mean (μ = {mu:.4f})')
    z3_thresh = mu + (z_threshold * sigma)
    plt.axvline(z3_thresh, color='orange', linestyle='dashed', linewidth=2, label=f'Z={z_threshold} (Threshold = {z3_thresh:.4f})')
    
    plt.title("Distribution of Neuron Expertise Scores (Log Scale)")
    plt.xlabel("Expertise Score (AP)")
    plt.ylabel("Number of Neurons (Log Scale)")
    plt.legend()
    
    # Save the plot to disk
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"expertise_histogram_z={z_threshold}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[+] Histogram saved successfully to: {plot_path}")
    print("======================================================\n")
    
    # Return the Z=3 threshold as a scientifically sound default
    return z3_thresh

if __name__ == '__main__':
    print("\n[ RUNNING COMPARISONS ]")
    
    paths = {
        "Baseline - PL only": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_baseline_pl_only.jsonl",
        "Baseline - ALL training": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_baseline_15_no_lora.jsonl",
        "Masked - TH: 0.6": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.6.jsonl",
        "Masked - TH: 0.65": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.65.jsonl",
        "Masked - TH: 0.7": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.7.jsonl",
        "Masked - TH: 0.75": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.75.jsonl",
        "Masked - TH: 0.8": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.8.jsonl",
        "Masked - TH: 0.85": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.85.jsonl",
        "Masked - TH: 0.9": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.9.jsonl",
        "Masked - TH: 0.13851979213253252": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.13851979213253252.jsonl",
        "Masked - TH: 0.17340149236254926": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.17340149236254926.jsonl",
    }
    # run_comparison_masked(
    #     "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_baseline_15_no_lora.jsonl",
    #     "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.9.jsonl",
    #     "Masked (TH: 0.9) vs Baseline (ALL training)"
    # )
    run_comparison_more_models(paths, description="ALL MASKED VARIANTS vs BASELINES")

    for threshold in [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.13851979213253252, 0.17340149236254926]:
        print(
            f"\n[+] Total neurons detected for TH={threshold}: ",
            count_detected_neurons(
            f"./results/benchmark_specific/checkpoints_15_no_lora/Qwen2.5-Coder-1.5B-Instruct-Continuous/new_dataset/mceval_hard_jsonl_top_benchmark_neurons_10000_{threshold}.json"
            )
        )