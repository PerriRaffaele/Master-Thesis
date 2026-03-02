import numpy as np
from tqdm import tqdm
from scipy.stats import entropy

def compute_lape_scores(layers, active_token_counts, total_token_counts):
    """Calculates the LAPE score (Entropy) for every neuron."""
    print("\nComputing LAPE scores...")
    lape_results = {}
    languages = list(active_token_counts.keys())
    
    for layer_name in tqdm(layers, desc="Scoring Layers"):
        # Calculate the raw probability (p) of activation for each language. Shape: [num_languages, num_neurons]
        prob_matrix = []
        for lang in languages:
            active = active_token_counts[lang][layer_name]
            total = total_token_counts[lang][layer_name]
            p = active / max(total, 1) 
            prob_matrix.append(p)
            
        prob_matrix = np.array(prob_matrix) 
        
        num_neurons = prob_matrix.shape[1]
        layer_results = []
        
        for neuron_idx in range(num_neurons):
            # Activation probabilities for this specific neuron across all languages e.g., [py_prob, java_prob, cpp_prob, js_prob]
            neuron_probs = prob_matrix[:, neuron_idx]
            
            sum_probs = np.sum(neuron_probs)
            if sum_probs == 0:
                continue
                
            # Normalize to create a probability distribution
            p_hat = neuron_probs / sum_probs
            
            # Calculate Shannon Entropy
            neuron_entropy = entropy(p_hat, base=len(languages)) # base=len(languages) so max entropy is exactly 1.0
            
            # Identify the dominant language
            dominant_lang_idx = np.argmax(p_hat)
            dominant_lang = languages[dominant_lang_idx]
            
            layer_results.append({
                "neuron_idx": int(neuron_idx),
                "entropy": float(neuron_entropy),
                "dominant_lang": dominant_lang,
                "py_prob": float(p_hat[languages.index("python")])
            })
            
        lape_results[layer_name] = layer_results
        
    return lape_results
