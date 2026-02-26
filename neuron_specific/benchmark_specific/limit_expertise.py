import numpy as np

def limit_expertise(expertise_scores, top_k=50):
    """
    Returns the indices of the top-K neurons for each layer.
    """
    top_neurons = {}
    
    for layer_name, ap_scores in expertise_scores.items():
        # np.argsort sorts ascending, so we take the last K and reverse them
        top_indices = np.argsort(ap_scores)[-top_k:][::-1]
        
        # Store a list of tuples: (neuron_index, ap_score)
        top_neurons[layer_name] = [(int(idx), float(ap_scores[idx])) for idx in top_indices]
        
    return top_neurons