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

def run_comparison_more_models(models_dict: dict, description="MULTIPLE MODELS", benchmark_name="UNKNOWN"):
    print("============================================================================================")
    print(f"MECHANISTIC INTERPRETABILITY: {description} REPORT FOR BENCHMARK {benchmark_name.upper()}")
    print("============================================================================================\n")
    
    # 1. Gather all metrics
    results = {}
    for model_name, path in models_dict.items():
        acc, tsed, total = calculate_metrics(path)
        if total > 0:
            if model_name.lower().startswith("masked"):
                # Get number of neurons masked
                neurons_masked = count_detected_neurons(
                    f"./results/benchmark_specific/checkpoints_no_lora/Qwen2.5-Coder-1.5B-Instruct-Continuous/new_dataset/mceval_hard_jsonl_top_benchmark_neurons_10000_{model_name.split('TH: ')[1].split(' - ')[0]}.json"
                )
                if neurons_masked == 0:
                    neurons_masked = count_detected_neurons(
                    f"./results/benchmark_specific/checkpoints_no_lora/Qwen2.5-Coder-1.5B-Instruct-Continuous_10/benchmark_only/pure_memorization_neurons_TH{model_name.split('TH: ')[1].split(' - ')[0]}_Z{model_name.split('Z: ')[1]}.json"
                    )

                results[model_name] = {'acc': acc, 'tsed': tsed, 'total': total, 'neurons_masked': neurons_masked}
            else:
                results[model_name] = {'acc': acc, 'tsed': tsed, 'total': total, 'neurons_masked': 0}
        else:
            print(f"[!] Warning: No valid data found for '{model_name}' at {path}\n")

    if len(results) < 2:
        print("Not enough valid models to compare. Need at least 2.")
        return

    model_names = list(results.keys())

    # 2. Print Absolute Metrics Table
    print("--- ABSOLUTE METRICS ---")
    header_str = f"{'Model Name':<45} | {'Accuracy (%)':<15} | {'TSED Score':<15} | {'Samples'} | {'Neurons Masked'}"
    print(header_str)
    print("-" * len(header_str))
    for name in model_names:
        metrics = results[name]
        print(f"{name:<45} | {metrics['acc']:>14.2f}% | {metrics['tsed']:>15.4f} | {metrics['total']:>7} | {metrics['neurons_masked']:>13}")
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

def analyze_pass_distribution(all_training_path: str, pl_only_path: str, benchmark_name: str):
    print("=====================================================================================")
    print("DETAILED MEMORIZATION & REGRESSION ANALYSIS FOR BENCHMARK: " + benchmark_name.upper())
    print("=====================================================================================\n")
    
    def get_passed_set(filepath):
        passed_tasks = set()
        all_tasks = set()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line)
                    task_id = row['task_id']
                    all_tasks.add(task_id)
                    if row['passed']:
                        passed_tasks.add(task_id)
            return passed_tasks, all_tasks
        except FileNotFoundError:
            print(f"[!] Error: Could not find file at {filepath}")
            return set(), set()

    # 1. Load the sets
    pl_passed, pl_all = get_passed_set(pl_only_path)
    all_passed, all_all = get_passed_set(all_training_path)
    
    total_tasks = len(all_all)
    if total_tasks == 0:
        return

    # 2. Calculate Intersections and Differences using Set Math
    passed_both = all_passed.intersection(pl_passed)
    memorized = all_passed - pl_passed  # In ALL, but not in PL
    regressed = pl_passed - all_passed  # In PL, but not in ALL
    failed_both = all_all - (all_passed.union(pl_passed))

    # 3. Print the Breakdown Matrix
    print(f"Total Tasks Evaluated: {total_tasks}\n")
    
    print(f"{'Category':<30} | {'Count':<6} | {'% of Total'}")
    print("-" * 55)
    print(f"{'1. Memorized (ALL Training)':<30} | {len(memorized):<6} | {(len(memorized)/total_tasks)*100:>5.2f}%")
    print(f"{'2. Solved (PL Only)':<30} | {len(regressed):<6} | {(len(regressed)/total_tasks)*100:>5.2f}%")
    print(f"{'3. Passed Both':<30} | {len(passed_both):<6} | {(len(passed_both)/total_tasks)*100:>5.2f}%")
    print(f"{'4. Failed Both':<30} | {len(failed_both):<6} | {(len(failed_both)/total_tasks)*100:>5.2f}%")
    print("-" * 55)
    
    # Check if the math adds up perfectly
    assert len(memorized) + len(regressed) + len(passed_both) + len(failed_both) == total_tasks
    
    # 4. Print the specific IDs for the interesting categories
    print("\n--- Tasks Memorized (Gained via Contamination) ---")
    print(sorted(list(memorized)))
    
    print("\n--- Tasks Solved (Gained via Contamination) ---")
    if len(regressed) == 0:
        print("None! The model didn't solve anything.")
    else:
        print(sorted(list(regressed)))
        
    return memorized, regressed, passed_both

def analyze_masked_retention(masked_models_dict: dict, memorized: set, regressed: set, passed_both: set):
    """
    Evaluates how masked models perform on strictly separated task subsets:
    Memorized (ALL only), PL Only (Regressed), and Robust (Passed Both).
    """
    print("\n=====================================================================================")
    print("ABLATION: SUBSET ANALYSIS")
    print("=====================================================================================\n")
    
    def get_passed_set(filepath):
        passed_tasks = set()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line)
                    if row.get('passed', False):
                        passed_tasks.add(row['task_id'])
            return passed_tasks
        except FileNotFoundError:
            print(f"[!] Error: Could not find file at {filepath}")
            return set()

    total_memorized = len(memorized)
    total_robust = len(passed_both)
    total_pl_only = len(regressed)  # Strictly the tasks ONLY passed by PL
    
    # Table Header
    header = f"{'Masked Model (Z-Score)':<38} | {'Memorized Retained':<20} | {'Robust Retained':<20} | {'PL Only Recovered':<20}"
    print(header)
    print("-" * len(header))
    
    if total_memorized == 0 and total_robust == 0 and total_pl_only == 0:
        print("No tasks in the sets to analyze.")
        return

    results_data = {}

    for model_name, path in masked_models_dict.items():
        masked_passed = get_passed_set(path)
        
        # 1. Memorized Retained (In ALL only) -> Target: 0% (Forget the contamination)
        retained_memorized = masked_passed.intersection(memorized)
        mem_pct = (len(retained_memorized) / total_memorized) * 100 if total_memorized > 0 else 0
        
        # 2. Robust Retained (In BOTH) -> Target: 100% (Keep baseline general knowledge)
        retained_robust = masked_passed.intersection(passed_both)
        rob_pct = (len(retained_robust) / total_robust) * 100 if total_robust > 0 else 0
        
        # 3. PL Only Recovered (In PL only) -> Target: Higher is better (Cure catastrophic forgetting)
        recovered_pl_only = masked_passed.intersection(regressed)
        pl_pct = (len(recovered_pl_only) / total_pl_only) * 100 if total_pl_only > 0 else 0
        
        # Format the strings for the table
        mem_str = f"{len(retained_memorized)}/{total_memorized} ({mem_pct:>5.1f}%)"
        rob_str = f"{len(retained_robust)}/{total_robust} ({rob_pct:>5.1f}%)"
        pl_str = f"{len(recovered_pl_only)}/{total_pl_only} ({pl_pct:>5.1f}%)"
        
        print(f"{model_name:<38} | {mem_str:<20} | {rob_str:<20} | {pl_str:<20}")
        
        results_data[model_name] = {
            "retained_memorized": retained_memorized,
            "retained_robust": retained_robust,
            "recovered_pl_only": recovered_pl_only
        }

    return results_data

def plot_accuracy_vs_threshold(paths_dict: dict, benchmark_name: str, output_dir="./results/"):
    print(f"======================================================")
    print(f"GENERATING ACCURACY PLOT FOR {benchmark_name.upper()}")
    print(f"======================================================\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    pl_acc = None
    all_acc = None
    masked_data = []

    # 1. Extract data from the paths dictionary
    for key, path in paths_dict.items():
        acc, _, total = calculate_metrics(path)
        if total == 0:
            continue
            
        if "PL only" in key:
            pl_acc = acc
        elif "ALL training" in key:
            all_acc = acc
        elif key.startswith("Masked"):
            # Extract threshold from the string
            th_str = key.split("TH: ")[1].split(" -")[0]
            threshold = float(th_str)
            
            # Reconstruct the path to the JSON to count neurons
            neuron_json_path = f"./results/benchmark_specific/checkpoints_no_lora/Qwen2.5-Coder-1.5B-Instruct-Continuous/new_dataset/mceval_hard_jsonl_top_benchmark_neurons_10000_{th_str}.json"
            neurons_masked = count_detected_neurons(neuron_json_path)
            
            masked_data.append({
                "threshold": threshold,
                "acc": acc,
                "neurons": neurons_masked
            })

    # 2. Sort by threshold ascending (Low threshold on Left, High threshold on Right)
    masked_data = sorted(masked_data, key=lambda x: x["threshold"])

    thresholds = [d["threshold"] for d in masked_data]
    accs = [d["acc"] for d in masked_data]
    neurons = [d["neurons"] for d in masked_data]

    # 3. Create the Plot
    plt.figure(figsize=(12, 7))
    
    # Draw Baseline Lines
    if all_acc is not None:
        plt.axhline(y=all_acc, color='red', linestyle='--', linewidth=2, label=f"Baseline: ALL Training ({all_acc:.2f}%)")
    if pl_acc is not None:
        plt.axhline(y=pl_acc, color='blue', linestyle='--', linewidth=2, label=f"Baseline: PL Only ({pl_acc:.2f}%)")

    # Draw the Masked Models line
    plt.plot(thresholds, accs, marker='o', linestyle='-', color='#8E44AD', linewidth=2.5, markersize=8, label="Masked Models")

    # Annotate points with the number of masked neurons
    for x, y, n in zip(thresholds, accs, neurons):
        plt.annotate(
            f"{n} neurons", 
            (x, y), 
            textcoords="offset points", 
            xytext=(0, 12),  # Shift text slightly above the dot
            ha='center',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.9)
        )

    # 4. Format the graph
    plt.title(f"Ablation Impact: Accuracy vs Expertise Threshold ({benchmark_name.upper()})", fontsize=14, fontweight='bold')
    plt.xlabel("Expertise Threshold (Lower Threshold = More Neurons Masked)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Moved legend out of the way of the low-threshold drop
    plt.legend(loc='lower right', fontsize=11) 
    
    # 5. Save the plot
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"accuracy_vs_threshold_{benchmark_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def split_benchmark_by_memorization(benchmark_filepath, memorized_task_ids):
    """
    Reads the raw benchmark file and splits it into two lists:
    1. Texts of tasks the model memorized.
    2. Texts of tasks the model did not memorize (to be used as the perfect control).
    """
    memorized_texts = []
    non_memorized_texts = []
    
    with open(benchmark_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            task_id = row['task_id']
            
            # Reconstruct the exact text used during continuous pre-training
            # (Adjust this if you only want the prompt, but usually contamination is prompt + solution)
            text = row.get('prompt', '') + row.get('canonical_solution', '')
            
            if task_id in memorized_task_ids:
                memorized_texts.append(text)
            else:
                non_memorized_texts.append(text)
                
    return memorized_texts, non_memorized_texts

def find_converged_checkpoint(base_dir: str, threshold: float = 0.02):
    """
    Loops through checkpoint directories numerically, calculates the average loss 
    for that specific epoch, and stops when the loss stops improving.
    
    Args:
        base_dir: The root folder containing all the checkpoint-XXX folders.
        threshold: The minimum relative improvement required (0.02 = 2%). 
                   If improvement is less than this, it declares convergence.
    """
    print(f"======================================================")
    print(f"SCANNING CHECKPOINTS FOR CONVERGENCE")
    print(f"======================================================\n")
    
    # 1. Find and sort all checkpoint folders by their step number
    checkpoint_dirs = []
    if not os.path.exists(base_dir):
        print(f"[!] Error: Directory {base_dir} does not exist.")
        return None
        
    for folder in os.listdir(base_dir):
        if folder.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, folder)):
            step = int(folder.split("-")[1])
            checkpoint_dirs.append((step, os.path.join(base_dir, folder)))
            
    # Sort them numerically (497, 994, 1491...)
    checkpoint_dirs.sort(key=lambda x: x[0])
    
    if not checkpoint_dirs:
        print(f"[!] No checkpoints found in {base_dir}")
        return None

    prev_epoch_loss = None
    
    print(f"{'Checkpoint':<20} | {'Avg Epoch Loss':<15} | {'Improvement'}")
    print("-" * 55)

    # 2. Loop through each checkpoint in order
    for step, ckpt_path in checkpoint_dirs:
        state_file = os.path.join(ckpt_path, "trainer_state.json")
        if not os.path.exists(state_file):
            continue
            
        with open(state_file, "r") as f:
            state = json.load(f)
            
        # The exact epoch this checkpoint represents (e.g., 1.0, 2.0)
        current_epoch = state.get("epoch", 0)
        
        # 3. Isolate the losses for ONLY this epoch
        epoch_losses = []
        for log in state.get("log_history", []):
            if "loss" in log:
                log_epoch = log.get("epoch", 0)
                # If the log belongs to the current epoch (e.g., between 0.0 and 1.0)
                if current_epoch - 1 < log_epoch <= current_epoch:
                    epoch_losses.append(log["loss"])
                    
        if not epoch_losses:
            continue
            
        # Calculate the average loss for the current epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Base case: Epoch 1
        if prev_epoch_loss is None:
            print(f"{os.path.basename(ckpt_path):<20} | {avg_loss:<15.4f} | N/A")
            prev_epoch_loss = avg_loss
            continue
            
        # 4. Check for convergence
        improvement = (prev_epoch_loss - avg_loss) / prev_epoch_loss
        improvement_str = f"{improvement * 100:+.2f}%"
        
        print(f"{os.path.basename(ckpt_path):<20} | {avg_loss:<15.4f} | {improvement_str}")
        
        # If the loss improved by less than the threshold (or got worse/negative)
        if improvement < threshold:
            print("-" * 55)
            print(f"\n[+] Loss converged at {os.path.basename(ckpt_path)}!")
            print(f"[+] The improvement ({improvement * 100:.2f}%) dropped below the {threshold * 100:.2f}% threshold.")
            return ckpt_path
            
        prev_epoch_loss = avg_loss

    print("-" * 55)
    print("\n[!] Looped through all checkpoints, but the loss never flatlined.")
    print("[!] Returning the final checkpoint.")
    return checkpoint_dirs[-1][1]
    

if __name__ == '__main__':
    
    # paths_mceval = {
    #     "Baseline - Original": "./results/Qwen2.5_Coder_1.5B_Instruct/mceval_hard/iter_1/result_baseline.jsonl",
    #     "Baseline - PL only": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_baseline_pl_only.jsonl",
    #     "Baseline - ALL training": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_baseline_15_no_lora.jsonl",
    #     "Baseline - ALL training 5": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous_5/mceval_hard/iter_1/result_baseline.jsonl",
    #     "Baseline - ALL training 7": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous_7/mceval_hard/iter_1/result_baseline.jsonl",
    #     "Baseline - ALL training 8": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous_8/mceval_hard/iter_1/result_baseline.jsonl",
    #     "Baseline - ALL training 9": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous_9/mceval_hard/iter_1/result_baseline.jsonl",
    #     "Baseline - ALL training 10": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous_10/mceval_hard/iter_1/result_baseline.jsonl",
    #     "Masked - TH: 0.13851979213253252 - Z: 3": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.13851979213253252.jsonl",
    #     "Masked - TH: 0.17340149236254926 - Z: 4": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.17340149236254926.jsonl",
    #     "Masked - TH: 0.208283192592566 - Z: 5": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.208283192592566.jsonl",
    #     "Masked - TH: 0.2431648928225828 - Z: 6": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.2431648928225828.jsonl",
    #     "Masked - TH: 0.27804659305259954 - Z: 7": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.27804659305259954.jsonl",
    #     "Masked - TH: 0.3129282932826163 - Z: 8": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.3129282932826163.jsonl",
    #     "Masked - TH: 0.34780999351263303 - Z: 9": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.34780999351263303.jsonl",
    #     "Masked - TH: 0.3826916937426498 - Z: 10": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_masked_no_lora_10000_0.3826916937426498.jsonl",
    #     "Masked (78) - TH: 0.3969268068394689 - Z: 1": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous_10/mceval_hard/iter_1/result_masked_no_lora_benchmark_only_0.3969268068394689.jsonl",
    #     "Masked (78) - TH: 0.4326374638154583 - Z: 2": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous_10/mceval_hard/iter_1/result_masked_no_lora_benchmark_only_0.4326374638154583.jsonl",
    #     "Masked (78) - TH: 0.504058777767437 - Z: 4": "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous_10/mceval_hard/iter_1/result_masked_no_lora_benchmark_only_0.504058777767437.jsonl",
    # }
    # run_comparison_more_models(paths_mceval, description="ALL MASKED VARIANTS vs BASELINES", benchmark_name="mceval_hard")

    # memorized, regressed, passed_both = analyze_pass_distribution(
    #     "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous_10/mceval_hard/iter_1/result_baseline.jsonl",
    #     "./results/Qwen2.5_Coder_1.5B_Instruct_Continuous/mceval_hard/iter_1/result_baseline_pl_only.jsonl",
    #     "mceval_hard"
    # )
    # plot_accuracy_vs_threshold(paths_mceval, benchmark_name="mceval_hard")

    target_dir = "./checkpoints_15_no_lora_pl_only"
    
    best_checkpoint = find_converged_checkpoint(target_dir, threshold=0.02)
    
    print(f"\nYour optimal PL Only model is located at: {best_checkpoint}")