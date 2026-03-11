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