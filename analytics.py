import json
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib_venn import venn2
import matplotlib.patches as mpatches

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

def run_comparison_models(models_dict: dict, description="MULTIPLE MODELS", benchmark_name="UNKNOWN"):
    print("==================================================================================================")
    print(f"MECHANISTIC INTERPRETABILITY: {description} REPORT FOR BENCHMARK {benchmark_name.upper()}")
    print("==================================================================================================\n")
    
    # 1. Gather all metrics
    results = {}
    for model_name, path in models_dict.items():
        acc, tsed, total = calculate_metrics(path)
        if total > 0:
            if model_name.lower().startswith("masked"):
                neurons_masked = 0
                
                # Safely extract variables using regex (defaults if not found)
                th_match = re.search(r"TH:\s*([\d\.]+)", model_name)
                z_match = re.search(r"Z:\s*([\d\.]+)", model_name)
                epoch_match = re.search(r"Epoch\s*(\d+)", model_name)
                
                th_str = th_match.group(1) if th_match else "UNKNOWN"
                z_str = z_match.group(1) if z_match else "UNKNOWN"
                # If Epoch isn't specified in the key, default to 10 based on earlier runs
                epoch_str = epoch_match.group(1) if epoch_match else "3" 

                # Define the potential paths where the JSON mask might live
                candidate_paths = [
                    # New Method (Benchmark Only) - with checkpoints folder
                    f"./results/benchmark_specific/checkpoints_no_lora_new/Qwen2.5-Coder-1.5B-Instruct-Continuous_{epoch_str}/benchmark_only/pure_memorization_neurons_TH{th_str}_Z{z_str}.json",

                    f"./results/benchmark_specific/checkpoints_with_2k_multi/Qwen2.5-Coder-1.5B-Instruct-Continuous_{epoch_str}/benchmark_only/original_pure_memorization_neurons_TH{th_str}_Z{z_str}.json",
                    
                    # New Method (Benchmark Only) - without checkpoints folder
                    f"./results/benchmark_specific/Qwen2.5-Coder-1.5B-Instruct-Continuous_{epoch_str}/benchmark_only/pure_memorization_neurons_TH{th_str}_Z{z_str}.json",
                    
                    # Old Method (The Stack Control)
                    f"./results/benchmark_specific/checkpoints_no_lora/Qwen2.5-Coder-1.5B-Instruct-Continuous_{epoch_str}/new_dataset/mceval_hard_jsonl_top_benchmark_neurons_10000_{th_str}_Z{z_str}.json",

                    f"./results/benchmark_specific/checkpoints_no_lora_new/Qwen2.5-Coder-1.5B-Instruct-Continuous_{epoch_str}/new_dataset/mceval_hard_jsonl_top_benchmark_neurons_10000_{th_str}_Z{z_str}.json",
                    
                    f"./results/benchmark_specific/checkpoints_with_2k_multi/Qwen2.5-Coder-1.5B-Instruct-Continuous_3/new_dataset/mceval_hard_jsonl_top_benchmark_neurons_10000_{th_str}_Z{z_str}.json"
                ]

                # Try each candidate path until one works
                for candidate in candidate_paths:
                    if os.path.exists(candidate):
                        neurons_masked = count_detected_neurons(candidate)
                        if neurons_masked > 0:
                            break # Found the file and counted neurons successfully!
                
                if neurons_masked == 0:
                    print(f"[!] Could not find or read a valid JSON mask for: {model_name}")

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
    z_thresh = mu + (z_threshold * sigma)
    plt.axvline(z_thresh, color='orange', linestyle='dashed', linewidth=2, label=f'Z={z_threshold} (Threshold = {z_thresh:.4f})')
    
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
    return z_thresh

def analyze_pass_distribution(all_training_path: str, pl_only_path: str, benchmark_name: str, model_id: str):
    print("\n=================================================================================================================")
    print("DETAILED MEMORIZATION & REGRESSION ANALYSIS FOR BENCHMARK: " + benchmark_name.upper() + " - " + model_id)
    print("=================================================================================================================")
    
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
    print(f"{'1. Leaked model':<30} | {len(memorized):<6} | {(len(memorized)/total_tasks)*100:>5.2f}%")
    print(f"{'2. PL only (2k multi)':<30} | {len(regressed):<6} | {(len(regressed)/total_tasks)*100:>5.2f}%")
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

def analyze_masked_retention(masked_models_dict: dict, memorized: set, regressed: set, passed_both: set, model_id: str):
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
    header = f"{'Masked Model (Z-Score)':<38} | {'Memorized Retained':<20} | {'Robust Retained':<20} | {'PL Only Recovered':<20} | {'Neurons Masked':<20}"
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
        
        # 3. PL Only Recovered (In PL only) -> Target: Higher is better
        recovered_pl_only = masked_passed.intersection(regressed)
        pl_pct = (len(recovered_pl_only) / total_pl_only) * 100 if total_pl_only > 0 else 0

        match = re.search(r'_(\d+\.\d+)_Z(\d+)\.jsonl', path)

        if match:
            # Extract the matched groups
            threshold, z = match.groups()
        else:
            print("No match found.")
        neuron_json_path = f"/home/raffaele/Thesis/results/benchmark_specific/checkpoints_with_2k_multi/{model_id}/benchmark_only/original_pure_memorization_neurons_TH{threshold}_Z{z}.json"
        neurons_masked = count_detected_neurons(neuron_json_path)


        # Format the strings for the table
        mem_str = f"{len(retained_memorized)}/{total_memorized} ({mem_pct:>5.1f}%)"
        rob_str = f"{len(retained_robust)}/{total_robust} ({rob_pct:>5.1f}%)"
        pl_str = f"{len(recovered_pl_only)}/{total_pl_only} ({pl_pct:>5.1f}%)"
        
        print(f"{model_name:<38} | {mem_str:<20} | {rob_str:<20} | {pl_str:<20} | {neurons_masked:<20}")
        
        results_data[model_name] = {
            "retained_memorized": retained_memorized,
            "retained_robust": retained_robust,
            "recovered_pl_only": recovered_pl_only,
            "neurons_masked": neurons_masked
        }

    return results_data

def plot_accuracy_vs_threshold(paths_dict: dict, benchmark_name: str, output_dir="./results/"):
    print(f"\n======================================================")
    print(f"GENERATING ACCURACY PLOT FOR {benchmark_name.upper()}")
    print(f"======================================================\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    pl_acc = None
    all_acc = None
    istr_acc = None
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
        elif "Original" in key:
            istr_acc = acc
        elif key.startswith("Masked"):
            # Extract threshold from the string
            th_str = key.split("TH: ")[1].split(" -")[0]
            threshold = float(th_str)
            z_str = key.split("Z: ")[1].split(")")[0]
            z = int(z_str)
            
            # Reconstruct the path to the JSON to count neurons
            neuron_json_path = f"./results/benchmark_specific/checkpoints_no_lora_new/Qwen2.5-Coder-1.5B-Instruct-Continuous_4/new_dataset/mceval_hard_jsonl_top_benchmark_neurons_10000_{th_str}_Z{z}.json"
            neurons_masked = count_detected_neurons(neuron_json_path)
            
            masked_data.append({
                "threshold": threshold,
                "acc": acc,
                "neurons": neurons_masked,
                "z": z
            })

    # 2. Sort by threshold ascending (Low threshold on Left, High threshold on Right)
    masked_data = sorted(masked_data, key=lambda x: x["threshold"])

    thresholds = [d["threshold"] for d in masked_data]
    accs = [d["acc"] for d in masked_data]
    neurons = [d["neurons"] for d in masked_data]
    zs = [d["z"] for d in masked_data]

    # 3. Create the Plot
    plt.figure(figsize=(12, 7))
    
    # Draw Baseline Lines
    if all_acc is not None:
        plt.axhline(y=all_acc, color='red', linestyle='--', linewidth=2, label=f"Baseline: ALL Training ({all_acc:.2f}%)")
    if pl_acc is not None:
        plt.axhline(y=pl_acc, color='blue', linestyle='--', linewidth=2, label=f"Baseline: PL Only ({pl_acc:.2f}%)")
    if istr_acc is not None:
        plt.axhline(y=istr_acc, color='green', linestyle='--', linewidth=2, label=f"Baseline: INSTRUCT ({istr_acc:.2f}%)")

    # Draw the Masked Models line
    plt.plot(thresholds, accs, marker='o', linestyle='-', color='#8E44AD', linewidth=2.5, markersize=8, label="Masked Models")

    # Annotate points with the number of masked neurons
    for x, y, n, z_val in zip(thresholds, accs, neurons, zs):
        plt.annotate(
            f"{n} neurons\n(z={z_val})", 
            (x, y), 
            textcoords="offset points", 
            xytext=(0, 15),  # Shift text slightly above the dot
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

def find_converged_checkpoint(base_dir: str, threshold: float = 0.02, model: str = "Unknown"):
    """
    Loops through checkpoint directories numerically, calculates the average loss 
    for that specific epoch, and stops when the loss stops improving.
    
    Args:
        base_dir: The root folder containing all the checkpoint-XXX folders.
        threshold: The minimum relative improvement required (0.02 = 2%). 
                   If improvement is less than this, it declares convergence.
    """
    print(f"======================================================")
    print(f"SCANNING CHECKPOINTS FOR CONVERGENCE - {model.upper()}")
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
        
    return checkpoint_dirs[-1][1]

def diff_and_intersect(pl_only_path: str, all_training_path: str, original_path: str, benchmark_name: str, model_id: str):
    print("\n=================================================================================================================")
    print("DETAILED MEMORIZATION & REGRESSION ANALYSIS FOR BENCHMARK: " + benchmark_name.upper() + " - " + model_id)
    print("=================================================================================================================")
    
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
    original_passed, original_all = get_passed_set(original_path)
    
    total_tasks = len(all_all)
    if total_tasks == 0:
        return

    # 2. Calculate passed tasks difference between Original and the Fine-Tuned models
    print(f"Original Passed Tasks:     {len(original_passed)}")
    print(f"PL Only Passed Tasks:      {len(pl_passed)}")
    print(f"ALL Training Passed Tasks: {len(all_passed)}\n")
    
    pl_original_diff = pl_passed - original_passed
    all_original_diff = all_passed - original_passed
    
    print(f"Delta PL Only:      {len(pl_original_diff)} new tasks learned.")
    print(f"Delta Leaked Training: {len(all_original_diff)} new tasks learned.")

    # 3. Calculate Intersections and Differences using Set Math
    passed_both = all_original_diff.intersection(pl_original_diff)
    memorized = all_original_diff - pl_original_diff  
    regressed = pl_original_diff - all_original_diff  

    # 4. Print the Breakdown Matrix
    # We calculate the total universe of *newly learned* tasks across both models
    total_new_learned = len(passed_both) + len(memorized) + len(regressed)
    
    print(f"\nTotal NEW Tasks Learned Across Both Models: {total_new_learned}")
    
    print("-" * 65)
    print(f"{'Category':<40} | {'Count':<6} | {'% of New Tasks'}")
    print("-" * 65)
    print(f"{'1. Pure Memorization (Leaked model)':<40} | {len(memorized):<6} | {(len(memorized)/total_new_learned)*100:>5.2f}%")
    print(f"{'2. Capacity Starved (PL Only)':<40} | {len(regressed):<6} | {(len(regressed)/total_new_learned)*100:>5.2f}%")
    print(f"{'3. Robust Generalization (Passed Both)':<40} | {len(passed_both):<6} | {(len(passed_both)/total_new_learned)*100:>5.2f}%")
    print("-" * 65)
    
    # 5. Print the specific IDs for the interesting categories
    print("\n--- Tasks Memorized (Gained exclusively via Contamination) ---")
    if len(memorized) == 0:
        print("None.")
    else:
        print(sorted(list(memorized)))
    
    print("\n--- Tasks Regressed ---")
    if len(regressed) == 0:
        print("None! The ALL model learned everything the PL model did.")
    else:
        print(sorted(list(regressed)))
        
    return memorized, regressed, passed_both

def analyze_pass_distribution_multi_iter(all_training_path_iter1: str, pl_only_path_iter1: str, benchmark_name: str, model_id: str, num_iters: int = 5):
    print("\n=================================================================================================================")
    print(f"STRICT {num_iters}-ITERATION MEMORIZATION & REGRESSION ANALYSIS: " + benchmark_name.upper() + " - " + model_id)
    print("=================================================================================================================")
    
    def get_consistent_passed_set(base_filepath):
        passed_in_all_iters = None
        all_tasks = set()
        
        for i in range(1, num_iters + 1):
            filepath = base_filepath.replace("iter_1", f"iter_{i}")
            current_iter_passed = set()
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        row = json.loads(line)
                        task_id = row['task_id']
                        all_tasks.add(task_id)
                        if row.get('passed', False):
                            current_iter_passed.add(task_id)
            except FileNotFoundError:
                print(f"[!] Error: Could not find file at {filepath}")
                return set(), all_tasks
            
            if passed_in_all_iters is None:
                passed_in_all_iters = current_iter_passed
            else:
                passed_in_all_iters = passed_in_all_iters.intersection(current_iter_passed)
                
        return passed_in_all_iters, all_tasks

    # 1. Load the STRICT sets
    pl_passed, pl_all = get_consistent_passed_set(pl_only_path_iter1)
    all_passed, all_all = get_consistent_passed_set(all_training_path_iter1)
    
    total_tasks = len(all_all)
    if total_tasks == 0:
        return set(), set(), set()

    # 2. Calculate Intersections and Differences using Set Math
    passed_both = all_passed.intersection(pl_passed)
    memorized = all_passed - pl_passed  
    regressed = pl_passed - all_passed  
    failed_both = all_all - (all_passed.union(pl_passed))

    # Calculate Percentages for plotting
    pct_mem = (len(memorized) / total_tasks) * 100
    pct_reg = (len(regressed) / total_tasks) * 100
    pct_both = (len(passed_both) / total_tasks) * 100
    pct_fail = (len(failed_both) / total_tasks) * 100

    # 3. Print the Breakdown Matrix
    print(f"Total Tasks Evaluated: {total_tasks}\n")
    print(f"{'Category (Passed ALL 5 Iters)':<35} | {'Count':<6} | {'% of Total'}")
    print("-" * 60)
    print(f"{'1. Leaked model':<35} | {len(memorized):<6} | {pct_mem:>5.2f}%")
    print(f"{'2. PL only (2k multi)':<35} | {len(regressed):<6} | {pct_reg:>5.2f}%")
    print(f"{'3. Passed Both':<35} | {len(passed_both):<6} | {pct_both:>5.2f}%")
    print(f"{'4. Failed/Flaky (Did not pass all)':<35} | {len(failed_both):<6} | {pct_fail:>5.2f}%")
    print("-" * 60)
    
    assert len(memorized) + len(regressed) + len(passed_both) + len(failed_both) == total_tasks
    
    # Safe string for filenames
    safe_model_id = model_id.replace("/", "-").replace(" ", "_")

    # =================================================================
    # DONUT CHART
    # =================================================================
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = ['Memorized\n(Leaked Only)', 'Regressed\n(PL Only)', 'Robust\n(Passed Both)', 'Failed\n(Failed Both)']
    sizes = [len(memorized), len(regressed), len(passed_both), len(failed_both)]
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#BDC3C7']
    
    # Filter out categories that are 0 so they don't break the plot
    filtered_labels = [l for l, s in zip(labels, sizes) if s > 0]
    filtered_sizes = [s for s in sizes if s > 0]
    filtered_colors = [c for c, s in zip(colors, sizes) if s > 0]

    # Create the pie chart
    wedges, texts, autotexts = ax.pie(
        filtered_sizes, labels=filtered_labels, colors=filtered_colors, 
        autopct='%1.1f%%', startangle=140, pctdistance=0.82, 
        textprops=dict(color="black", fontweight='bold', fontsize=11)
    )
    
    # Draw a white circle in the center to turn it into a Donut Chart
    centre_circle = plt.Circle((0,0), 0.65, fc='white')
    fig.gca().add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')  
    plt.title(f"Task Outcomes\n{benchmark_name.upper()} - {model_id}", fontsize=14, fontweight='bold', pad=20)
    
    # Add the total number of tasks in the very center of the donut
    plt.text(0, 0, f"Total Tasks\n{total_tasks}", ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.savefig(f"./results/donut_{benchmark_name}_{safe_model_id}.png", dpi=300, bbox_inches='tight')
    plt.close()
        
    return memorized, regressed, passed_both

def diff_and_intersect_multi_iter(pl_only_path: str, all_training_path: str, original_path: str, benchmark_name: str, model_id: str, num_iters: int = 5):
    print("\n=================================================================================================================")
    print("DETAILED MEMORIZATION & REGRESSION ANALYSIS FOR BENCHMARK: " + benchmark_name.upper() + " - " + model_id)
    print("=================================================================================================================")
    
    def get_passed_set(base_filepath):
        passed_in_all_iters = None
        all_tasks = set()
        
        # Loop through iter_1, iter_2, ..., iter_5
        for i in range(1, num_iters + 1):
            # Dynamically replace 'iter_1' in the string with the current iteration
            filepath = base_filepath.replace("iter_1", f"iter_{i}")
            current_iter_passed = set()
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        row = json.loads(line)
                        task_id = row['task_id']
                        all_tasks.add(task_id)
                        if row.get('passed', False):
                            current_iter_passed.add(task_id)
            except FileNotFoundError:
                print(f"[!] Error: Could not find file at {filepath}")
                # If a file is missing, we can't guarantee it passed 5 times.
                return set(), all_tasks
            
            # If it's the first iteration, initialize the set
            if passed_in_all_iters is None:
                passed_in_all_iters = current_iter_passed
            else:
                # Keep ONLY tasks that also passed in this current iteration
                passed_in_all_iters = passed_in_all_iters.intersection(current_iter_passed)
                
        return passed_in_all_iters, all_tasks

    # 1. Load the sets
    pl_passed, pl_all = get_passed_set(pl_only_path)
    all_passed, all_all = get_passed_set(all_training_path)
    original_passed, original_all = get_passed_set(original_path)
    failed_both = all_all - (all_passed.union(pl_passed))

    
    total_tasks = len(all_all)
    if total_tasks == 0:
        return

    # 2. Calculate passed tasks difference between Original and the Fine-Tuned models
    print(f"Original Passed Tasks:     {len(original_passed)}")
    print(f"PL Only Passed Tasks:      {len(pl_passed)}")
    print(f"ALL Training Passed Tasks: {len(all_passed)}\n")
    
    pl_original_diff = pl_passed - original_passed
    all_original_diff = all_passed - original_passed
    
    print(f"Delta PL Only:      {len(pl_original_diff)} new tasks learned.")
    print(f"Delta Leaked Training: {len(all_original_diff)} new tasks learned.")

    # 3. Calculate Intersections and Differences using Set Math
    passed_both = all_original_diff.intersection(pl_original_diff)
    memorized = all_original_diff - pl_original_diff  
    regressed = pl_original_diff - all_original_diff  

    # 4. Print the Breakdown Matrix
    # We calculate the total universe of *newly learned* tasks across both models
    total_new_learned = len(passed_both) + len(memorized) + len(regressed)
    
    print(f"\nTotal NEW Tasks Learned Across Both Models: {total_new_learned}")
    
    print("-" * 65)
    print(f"{'Category':<40} | {'Count':<6} | {'% of New Tasks'}")
    print("-" * 65)
    print(f"{'1. Pure Memorization (Leaked model)':<40} | {len(memorized):<6} | {(len(memorized)/total_new_learned)*100:>5.2f}%")
    print(f"{'2. Capacity Starved (PL Only)':<40} | {len(regressed):<6} | {(len(regressed)/total_new_learned)*100:>5.2f}%")
    print(f"{'3. Robust Generalization (Passed Both)':<40} | {len(passed_both):<6} | {(len(passed_both)/total_new_learned)*100:>5.2f}%")
    print("-" * 65)

    # =================================================================
    # DONUT CHART
    # =================================================================
    safe_model_id = model_id.replace("/", "-").replace(" ", "_")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = ['Memorized\n(Leaked Only)', 'Regressed\n(PL Only)', 'Robust\n(Passed Both)', 'Failed\n(Failed Both)']
    sizes = [len(memorized), len(regressed), len(passed_both), len(failed_both)]
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#BDC3C7']
    
    # Filter out categories that are 0 so they don't break the plot
    filtered_labels = [l for l, s in zip(labels, sizes) if s > 0]
    filtered_sizes = [s for s in sizes if s > 0]
    filtered_colors = [c for c, s in zip(colors, sizes) if s > 0]

    # Create the pie chart
    wedges, texts, autotexts = ax.pie(
        filtered_sizes, labels=filtered_labels, colors=filtered_colors, 
        autopct='%1.1f%%', startangle=140, pctdistance=0.82, 
        textprops=dict(color="black", fontweight='bold', fontsize=11)
    )
    
    # Draw a white circle in the center to turn it into a Donut Chart
    centre_circle = plt.Circle((0,0), 0.65, fc='white')
    fig.gca().add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')  
    plt.title(f"Task Outcomes Without Instruct\n{benchmark_name.upper()} - {model_id}", fontsize=14, fontweight='bold', pad=20)
    
    # Add the total number of tasks in the very center of the donut
    plt.text(0, 0, f"Total Tasks\n{total_tasks}", ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.savefig(f"./results/donut_no_istruct_{benchmark_name}_{safe_model_id}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Print the specific IDs for the interesting categories
    print("\n--- Tasks Memorized (Gained exclusively via Contamination) ---")
    if len(memorized) == 0:
        print("None.")
    else:
        print(sorted(list(memorized)))
    
    print("\n--- Tasks Regressed ---")
    if len(regressed) == 0:
        print("None! The ALL model learned everything the PL model did.")
    else:
        print(sorted(list(regressed)))
        
    return memorized, regressed, passed_both

def get_passed_set_acc(base_filepath, num_iters: int = 5):
    passed_in_all_iters = None
    all_tasks = set()
    
    # Loop through iter_1, iter_2, ..., iter_5
    for i in range(1, num_iters + 1):
        # Dynamically replace 'iter_1' in the string with the current iteration
        filepath = base_filepath.replace("iter_1", f"iter_{i}")
        current_iter_passed = set()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line)
                    task_id = row['task_id']
                    all_tasks.add(task_id)
                    if row.get('passed', False):
                        current_iter_passed.add(task_id)
        except FileNotFoundError:
            print(f"[!] Error: Could not find file at {filepath}")
            # If a file is missing, we can't guarantee it passed all times. Return 0.0 accuracy.
            return set(), all_tasks, 0.0
        
        # If it's the first iteration, initialize the set
        if passed_in_all_iters is None:
            passed_in_all_iters = current_iter_passed
        else:
            # Keep ONLY tasks that also passed in this current iteration
            passed_in_all_iters = passed_in_all_iters.intersection(current_iter_passed)
            
    # Calculate Accuracy
    if len(all_tasks) == 0:
        accuracy = 0.0
    else:
        accuracy = (len(passed_in_all_iters) / len(all_tasks)) * 100
            
    return passed_in_all_iters, all_tasks, accuracy


if __name__ == '__main__':

    # target_dir = "./checkpoints_multi_language_2k"
    
    # best_checkpoint = find_converged_checkpoint(target_dir, threshold=0.02, model="PL Only")


    # target_dir = "./checkpoints_with_2k_multi"
    
    # best_checkpoint = find_converged_checkpoint(target_dir, threshold=0.02, model="ALL Training New")

    paths_mceval = {
        "Original Instruct": "./results/instruct/Qwen2.5_Coder_1.5B_Instruct/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "Baseline - NEW - PL ONLY - 2k multi - Epoch 2": "./results/2k_new_training_multi_language/Qwen2.5_Coder_1.5B_Instruct_Continuous_2/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "Baseline - NEW -Epoch 3 multi": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "Masked - TH: 0.11177004401792183 - Z: 2": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.11177004401792183_Z2.jsonl",
        "Masked - TH: 0.15027412740521853 - Z: 3": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.15027412740521853_Z3.jsonl",
        "Masked - TH: 0.18877821079251522 - Z: 4": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.18877821079251522_Z4.jsonl",
        "Masked - TH: 0.2272822941798119 - Z: 5": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.2272822941798119_Z5.jsonl",
        "Masked - TH: 0.26578637756710866 - Z: 6": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.26578637756710866_Z6.jsonl",
        "Masked - TH: 0.30429046095440526 - Z: 7": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.30429046095440526_Z7.jsonl",
        "Masked - TH: 0.342794544341702 - Z: 8": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.342794544341702_Z8.jsonl",
        "Masked - OLD TH: 0.30429046095440526 - Z: 2": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.11177004401792183_Z2_old.jsonl",
        "Masked - OLD TH: 0.30429046095440526 - Z: 3": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.15027412740521853_Z3_old.jsonl",
        "Masked - OLD TH: 0.30429046095440526 - Z: 4": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.18877821079251522_Z4_old.jsonl",
        "Masked - OLD TH: 0.30429046095440526 - Z: 5": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.2272822941798119_Z5_old.jsonl",
        "Masked - OLD TH: 0.30429046095440526 - Z: 6": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.26578637756710866_Z6_old.jsonl",
        "Masked - OLD TH: 0.30429046095440526 - Z: 7": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.30429046095440526_Z7_old.jsonl",
        "Masked - OLD TH: 0.342794544341702 - Z: 8": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.342794544341702_Z8_old.jsonl",
        # "Masked Pure - TH: 0.2347929855128098 - Z: 1":"./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.2347929855128098_Z1.jsonl",
        # "Masked Pure - TH: 0.28712553574286115 - Z: 2":"./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.28712553574286115_Z2.jsonl",
        # "Masked Pure - TH: 0.33945808597291244 - Z: 3":"./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.33945808597291244_Z3.jsonl",
        # "Masked Pure - TH: 0.3917906362029637 - Z: 4":"./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.3917906362029637_Z4.jsonl",
        # "Masked Pure - TH: 0.44412318643301507 - Z: 5":"./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.44412318643301507_Z5.jsonl",
        # "Masked Pure - TH: 0.4964557366630664 - Z: 6":"./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.4964557366630664_Z6.jsonl",
        # "Masked Pure - TH: 0.5487882868931176 - Z: 7":"./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.5487882868931176_Z7.jsonl",
        # "Masked Pure - TH: 0.601120837123169 - Z: 8":"./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.601120837123169_Z8.jsonl",
    }
    run_comparison_models(paths_mceval, description="ALL MASKED VARIANTS vs BASELINES", benchmark_name="mceval_hard")

    memorized_1, regressed_1, passed_both_1 = analyze_pass_distribution(
        "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "./results/2k_new_training_multi_language/Qwen2.5_Coder_1.5B_Instruct_Continuous_2/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "mceval_hard",
        "Qwen2.5-Coder-1.5B-Instruct-Continuous_3 - Not masked"
    )

    memorized_new, regressed_new, passed_both_new = diff_and_intersect(
        "./results/2k_new_training_multi_language/Qwen2.5_Coder_1.5B_Instruct_Continuous_2/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "./results/instruct/Qwen2.5_Coder_1.5B_Instruct/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "mceval_hard",
        "Qwen2.5-Coder-1.5B-Instruct_Continuous_3"
    )

    # 1. Calculate the intersections
    regressed_overlap = regressed_1.intersection(regressed_new)
    memorized_overlap = memorized_1.intersection(memorized_new)
    
    # 2. Print the results clearly
    print("\n--- Overlap Analysis ---")
    print(f"Regressed Overlap: {len(regressed_overlap)} tasks are in both sets.")
    print(f" -> (This means {len(regressed_1) - len(regressed_overlap)} regressed tasks were already solved by the Original model)")
    
    print(f"Memorized Overlap: {len(memorized_overlap)} tasks are in both sets.")
    print(f" -> (This means {len(memorized_1) - len(memorized_overlap)} memorized tasks were already solved by the Original model)")
    
    paths = {
        "Masked - Pure Memorization (Z=1)": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.2347929855128098_Z1.jsonl",
        "Masked - Pure Memorization (Z=2)": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.28712553574286115_Z2.jsonl",
        "Masked - Pure Memorization (Z=3)": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.33945808597291244_Z3.jsonl",
        "Masked - Pure Memorization (Z=4)": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.3917906362029637_Z4.jsonl",
        "Masked - Pure Memorization (Z=5)": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.44412318643301507_Z5.jsonl",
        "Masked - Pure Memorization (Z=6)": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.4964557366630664_Z6.jsonl",
        "Masked - Pure Memorization (Z=7)": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.5487882868931176_Z7.jsonl",
        "Masked - Pure Memorization (Z=8)": "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_pure_memorization_0.601120837123169_Z8.jsonl",
    }
    results = analyze_masked_retention(paths, memorized_1, regressed_1, passed_both_1, "Qwen2.5-Coder-1.5B-Instruct-Continuous_3")

    for model_name, data in paths.items():
        overlap_pl_only_recovered = regressed_overlap.intersection(results[model_name]["recovered_pl_only"])
        print(f"\n--- Overlap between Regressed Tasks and PL Only Recovered Tasks in {model_name} ---")
        print(f"{len(overlap_pl_only_recovered)} tasks are in both sets.")


    memorized, regressed, passed_both = analyze_pass_distribution(
        "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_masked_0.30429046095440526_Z7.jsonl",
        "./results/2k_new_training_multi_language/Qwen2.5_Coder_1.5B_Instruct_Continuous_2/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "mceval_hard",
        "Qwen2.5-Coder-1.5B-Instruct-Continuous_3 - Masked Z=7 (TH: 0.30429046095440526)"
    )

    memorized_2, regressed_2, passed_both_2 = analyze_pass_distribution_multi_iter(
        "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "./results/2k_new_training_multi_language/Qwen2.5_Coder_1.5B_Instruct_Continuous_2/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "mceval_hard",
        "Qwen2.5-Coder-1.5B-Instruct-Continuous_3 - Not masked",
        num_iters=5
    )

    memorized_new_2, regressed_new_2, passed_both_new_2 = diff_and_intersect_multi_iter(
        "./results/2k_new_training_multi_language/Qwen2.5_Coder_1.5B_Instruct_Continuous_2/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "./results/instruct/Qwen2.5_Coder_1.5B_Instruct/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl",
        "mceval_hard",
        "Qwen2.5-Coder-1.5B-Instruct_Continuous_3",
        num_iters=5
    )

    passed_tasks, _, acc = get_passed_set_acc("./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl", num_iters=5)
    print(f"\nTasks (MCEVAL) that passed in ALL 5 iterations for the Leaked model: {len(passed_tasks)}, with accuracy: {acc:.2f}%") 
    passed_tasks, _, acc = get_passed_set_acc("./results/2k_new_training_multi_language/Qwen2.5_Coder_1.5B_Instruct_Continuous_2/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl", num_iters=5)
    print(f"Tasks (MCEVAL) that passed in ALL 5 iterations for the PL Only model: {len(passed_tasks)}, with accuracy: {acc:.2f}%")
    passed_tasks, _, acc = get_passed_set_acc("./results/instruct/Qwen2.5_Coder_1.5B_Instruct/mceval_hard/iter_1/result_baseline_mceval_hard.jsonl", num_iters=5)
    print(f"Tasks (MCEVAL) that passed in ALL 5 iterations for the Original model: {len(passed_tasks)}, with accuracy: {acc:.2f}%")

    passed_tasks, _, acc = get_passed_set_acc("./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/humaneval_plus/iter_1/result_baseline_humaneval_plus.jsonl", num_iters=5)
    print(f"\nTasks (HUMANEVAL) that passed in ALL 5 iterations for the Leaked model: {len(passed_tasks)}, with accuracy: {acc:.2f}%") 
    passed_tasks, _, acc = get_passed_set_acc("./results/instruct/Qwen2.5_Coder_1.5B_Instruct/humaneval_plus/iter_1/result_baseline_humaneval_plus.jsonl", num_iters=5)
    print(f"Tasks (HUMANEVAL) that passed in ALL 5 iterations for the Original model: {len(passed_tasks)}, with accuracy: {acc:.2f}%")

    passed_tasks, _, acc = get_passed_set_acc("./results/leakage_with_2k_multi/Qwen2.5_Coder_1.5B_Instruct_Continuous_3/mbpp_plus/iter_1/result_baseline_mbpp_plus.jsonl", num_iters=5)
    print(f"\nTasks (MBPP) that passed in ALL 5 iterations for the Leaked model: {len(passed_tasks)}, with accuracy: {acc:.2f}%") 
    passed_tasks, _, acc = get_passed_set_acc("./results/instruct/Qwen2.5_Coder_1.5B_Instruct/mbpp_plus/iter_1/result_baseline_mbpp_plus.jsonl", num_iters=5)
    print(f"Tasks (MBPP) that passed in ALL 5 iterations for the Original model: {len(passed_tasks)}, with accuracy: {acc:.2f}%")
