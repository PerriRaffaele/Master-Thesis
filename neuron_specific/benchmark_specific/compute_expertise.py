from sklearn.metrics import average_precision_score
import numpy as np
from tqdm import tqdm
import torch
import gc

def compute_expertise(target_acts, background_acts):
    """
    Calculates the AP score for every neuron in every layer.
    """
    print("Calculating AP scores for each neuron...")
    expertise_scores = {}
    
    # Create labels: 1 for Target (Benchmark), 0 for Background
    num_benchmark = len(target_acts[list(target_acts.keys())[0]]) 
    num_background = len(background_acts[list(background_acts.keys())[0]])
    labels = np.concatenate([
        np.ones(num_benchmark),
        np.zeros(num_background)
    ])
    
    for layer_name in tqdm(target_acts.keys(), desc="Scoring Layers"):
        # Combine activations: shape [total_samples, num_neurons]
        X = np.concatenate([target_acts[layer_name], background_acts[layer_name]], axis=0)
        num_neurons = X.shape[1]
        
        ap_scores = np.zeros(num_neurons)
        for neuron_idx in range(num_neurons):
            # Calculate AP for this specific neuron
            neuron_activations = X[:, neuron_idx]
            ap_scores[neuron_idx] = average_precision_score(labels, neuron_activations)
            
        expertise_scores[layer_name] = ap_scores
        
    return expertise_scores

def compute_expertise_chunked(target_chunk_paths, control_chunk_paths, layer_names):
    """
    Computes AP scores by loading data one layer at a time from disk to save RAM.
    """
    ap_scores_per_layer = {}
    
    print("\nCalculating AP scores layer by layer...")
    for layer in tqdm(layer_names, desc="Layers"):
        
        # 1. Gather all Target acts for THIS layer only
        target_acts_layer = []
        for path in target_chunk_paths:
            chunk_data = torch.load(path)
            target_acts_layer.append(chunk_data[layer])
            del chunk_data # Free memory immediately
        target_acts_layer = np.concatenate(target_acts_layer, axis=0) 
        
        # 2. Gather all Control acts for THIS layer only
        control_acts_layer = []
        for path in control_chunk_paths:
            chunk_data = torch.load(path)
            control_acts_layer.append(chunk_data[layer])
            del chunk_data 
        control_acts_layer = np.concatenate(control_acts_layer, axis=0) 
        
        # 3. Create Binary Labels (1 for Target, 0 for Control)
        y_true = np.concatenate([
            np.ones(target_acts_layer.shape[0]), 
            np.zeros(control_acts_layer.shape[0])
        ])
        
        # 4. Concatenate Activations
        y_scores = np.concatenate([target_acts_layer, control_acts_layer], axis=0)
        
        # 5. Compute AP for every neuron in this layer
        num_neurons = y_scores.shape[1]
        layer_ap_scores = []
        
        # You can also wrap this in tqdm if you want to see neuron-level progress
        for neuron_idx in range(num_neurons):
            ap = average_precision_score(y_true, y_scores[:, neuron_idx])
            layer_ap_scores.append(ap)
            
        ap_scores_per_layer[layer] = layer_ap_scores
        
        # 6. THE CRITICAL STEP: Empty RAM before the next layer!
        del target_acts_layer, control_acts_layer, y_true, y_scores
        gc.collect() 
        
    return ap_scores_per_layer