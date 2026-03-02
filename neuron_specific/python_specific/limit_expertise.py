def limit_python_expertise(lape_results, max_entropy=0.10):
    """
    Filters for neurons where Python is dominant AND entropy is extremely low.
    max_entropy=0.10 means it is highly exclusive to one language.
    """
    top_python_neurons = {}
    total_kept = 0
    
    for layer_name, neurons in lape_results.items():
        python_experts = []
        
        for n in neurons:
            if n["dominant_lang"] == "python" and n["entropy"] < max_entropy:
                python_experts.append([n["neuron_idx"], n["entropy"]])
                
        # Sort so the lowest entropy (most exclusive) is first
        python_experts.sort(key=lambda x: x[1])
        top_python_neurons[layer_name] = python_experts
        total_kept += len(python_experts)
        print(f"  {layer_name}: Kept {len(python_experts)} Python neurons")
        
    print(f"\nTotal Python-specific neurons found across all layers: {total_kept}")
    return top_python_neurons