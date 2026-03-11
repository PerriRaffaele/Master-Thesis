import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os
import gc

def get_mlp_hook(layer_name, activations_dict):
    def hook(module, args):
        # grab matrix of neuron activations for this layer (shape: [batch_size, seq_len, hidden_dim])
        hidden_states = args[0]
        max_act_over_seq, _ = torch.max(hidden_states, dim=1)  # [batch_size, hidden_dim]
        for act in max_act_over_seq:
            activations_dict[layer_name].append(act.cpu().numpy())
    return hook

# 3. Function to process a dataset
def compute_responses(model, tokenizer, activations_dict, texts, desc="Processing", batch_size=8):
    activations_dict.clear() # Reset for new run
    
    for text in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[text:text+batch_size]
        
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True,
            pad_to_multiple_of=8  # For better GPU efficiency
        ).to(model.device)
        
        with torch.no_grad():
            # We just do a forward pass to trigger the hooks, no need to generate
            model(**inputs)

        del inputs  # Free memory
        torch.cuda.empty_cache()  # Clear GPU memory after each batch to prevent OOM

    # Return dictionary of stacked numpy arrays
    return {layer: np.stack(acts) for layer, acts in activations_dict.items()}