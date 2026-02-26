import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def get_mlp_hook(layer_name, activations_dict):
    def hook(module, args):
        # In models like Qwen, the intermediate activations are computed before the final down_proj. 
        # We take the max activation across the sequence tokens to represent the whole sequence.
        # output shape is [batch, seq_len, intermediate_size]
        hidden_states = args[0] # args is a tuple, first element is the input tensor
        # We take the max activation across all tokens in the sequence (dim=1)
        # We also squeeze the batch dimension since we process batch_size=1
        max_act_over_seq, _ = torch.max(hidden_states.squeeze(0).detach(), dim=0) 
        
        # Move to CPU immediately to prevent GPU Out-Of-Memory errors
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