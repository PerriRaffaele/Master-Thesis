import numpy as np

def limit_expertise(expertise_scores, threshold=0.90):
    """
    Returns the indices of the top-K neurons for each layer.
    """
    top_neurons = {}
    total_kept = 0
    
    for layer_name, ap_scores in expertise_scores.items():
        # Find the indices of all neurons that pass the threshold
        passing_indices = np.where(ap_scores > threshold)[0]
        
        # Sort these passing neurons from highest score to lowest
        sorted_relative_indices = np.argsort(ap_scores[passing_indices])[::-1]
        
        # Map the relative sorting back to the actual neuron indices
        sorted_passing_indices = passing_indices[sorted_relative_indices]
        
        # Store them as (neuron_idx, ap_score) pairs for this layer
        layer_experts = [(int(idx), float(ap_scores[idx])) for idx in sorted_passing_indices]
        print(f"Layer {layer_name}: Keeping {len(layer_experts)} neurons with AP > {threshold}")
        top_neurons[layer_name] = layer_experts
        total_kept += len(layer_experts)
        
    print(f"\nTotal neurons kept across all layers: {total_kept}")
    return top_neurons