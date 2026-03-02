from tqdm import tqdm
import numpy as np
import torch

def get_lape_hook(layer_name, current_language, active_token_counts, total_token_counts):
    """
    Instead of finding the 'max' activation, this hook counts HOW MANY 
    tokens caused the neuron to fire (activation > 0).
    """
    def hook(module, args):
        hidden_states = args[0].squeeze(0).detach() # Shape: [seq_len, intermediate_size]
        
        # Number of tokens in sequence
        seq_len = hidden_states.shape[0]
        
        # How many tokens had an activation strictly > 0 for each neuron
        active_tokens = (hidden_states > 0).float().sum(dim=0).cpu().numpy() # hidden_states > 0 creates a boolean tensor and .sum(dim=0) counts True values per column.
        
        # Accumulate the totals for this specific layer and language
        if layer_name not in active_token_counts[current_language]:
            num_neurons = hidden_states.shape[1]
            active_token_counts[current_language][layer_name] = np.zeros(num_neurons)
            total_token_counts[current_language][layer_name] = 0
            
        active_token_counts[current_language][layer_name] += active_tokens
        total_token_counts[current_language][layer_name] += seq_len
        
    return hook

def compute_language_responses(model, tokenizer, multilingual_texts, active_token_counts, total_token_counts):
    """Runs texts through the model for each language."""
    active_token_counts.clear()
    total_token_counts.clear()
    
    hooks = []
    
    for lang, texts in multilingual_texts.items():
        print(f"\nProcessing {lang} dataset...")
        
        # Attach hooks specifically for this language
        for i, layer in enumerate(model.model.layers):
            h = layer.mlp.down_proj.register_forward_pre_hook(get_lape_hook(f"layer_{i}", lang, active_token_counts, total_token_counts))
            hooks.append(h)
            
        for text in tqdm(texts, desc=f"Forward Passes ({lang})"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                model(**inputs)
                
        # Remove hooks before moving to the next language
        for h in hooks: h.remove()
        hooks.clear()