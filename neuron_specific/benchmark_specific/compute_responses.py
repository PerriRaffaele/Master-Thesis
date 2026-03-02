import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def get_mlp_hook(layer_name, activations_dict):
    def hook(module, args):
        # grab matrix of neuron activations for this layer (shape: [batch_size, seq_len, hidden_dim])
        hidden_states = args[0]
        max_act_over_seq, _ = torch.max(hidden_states.squeeze(0).detach(), dim=0) 
        activations_dict[layer_name].append(max_act_over_seq.cpu().numpy())
    return hook

# 3. Function to process a dataset
def compute_responses(model, tokenizer, activations_dict, texts, desc="Processing"):
    activations_dict.clear() # Reset for new run
    for text in tqdm(texts, desc=desc):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            # We just do a forward pass to trigger the hooks, no need to generate
            model(**inputs)
    
    # Return dictionary of stacked numpy arrays
    return {layer: np.stack(acts) for layer, acts in activations_dict.items()}