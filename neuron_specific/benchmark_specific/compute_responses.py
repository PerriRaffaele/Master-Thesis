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

def compute_responses_chunks(model, tokenizer, activations_dict, texts, desc="Processing", batch_size=8, type="Target"):
    activations_dict.clear() # Reset for new run
    output_dir = './chunks/'
    chunk_size = 100000
    
    os.makedirs(output_dir, exist_ok=True)
    chunk_filepaths = []
    total_samples = len(texts)
    
    # 1. THE CHUNKING LOOP
    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        
        # Take the raw JSON strings exactly as they are
        current_chunk_texts = texts[chunk_start:chunk_end] 
        
        print(f"\n--- Processing Chunk: {chunk_start} to {chunk_end} ---")
        
        # 2. Process the chunk in batches
        for i in tqdm(range(0, len(current_chunk_texts), batch_size), desc=f"Batches"):
            batch = current_chunk_texts[i : i + batch_size]
            
            inputs = tokenizer(
                batch, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512, 
                padding=True,
                pad_to_multiple_of=8
            ).to(model.device)
            
            with torch.no_grad():
                model(**inputs)
                
            del inputs

        # 3. Stack this specific chunk into numpy arrays
        chunk_dict = {layer: np.stack(acts) for layer, acts in activations_dict.items()}
        
        # 4. Save the chunk to the hard drive!
        chunk_filename = os.path.join(output_dir, f"{desc.replace(' ', '_')}_{type}_chunk_{chunk_start}.pt")
        torch.save(chunk_dict, chunk_filename)
        chunk_filepaths.append(chunk_filename)
        
        # 5. Empty the RAM so the server doesn't crash!
        activations_dict.clear() 
        del chunk_dict           
        gc.collect()             
        
    # Return the list of file paths saved to disk
    return chunk_filepaths