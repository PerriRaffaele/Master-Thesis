"""
Adapted from the official repository: https://github.com/pjlintw/finetuning-transfer/
"""
from tqdm import tqdm
from peft import PeftModel
from torch.nn import Parameter
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc, fire, torch


def load_causal_lm_from_pretrained(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    )
    return model


def update_embed_tokens_and_lm_head(base_model, fine_tuned_model):
    device = base_model.model.embed_tokens.weight.device
    fine_tuned_model.to(device)

    # Update embed_tokens
    base_embed_weight = base_model.model.embed_tokens.weight
    fine_tuned_embed_weight = fine_tuned_model.model.embed_tokens.weight
    row_diff = fine_tuned_embed_weight.shape[0] - base_embed_weight.shape[0]
    if row_diff > 0:
        zero_padding = torch.zeros(row_diff, base_embed_weight.size(1), dtype=base_embed_weight.dtype, device=base_embed_weight.device)
        base_model.model.embed_tokens.weight = Parameter(torch.cat((base_embed_weight, zero_padding), dim=0))
    else:
        base_model.model.embed_tokens.weight = Parameter(base_embed_weight)

    # Update lm_head
    base_lm_head_weight = base_model.lm_head.weight
    fine_tuned_lm_head_weight = fine_tuned_model.lm_head.weight
    row_diff = fine_tuned_lm_head_weight.shape[0] - base_lm_head_weight.shape[0]
    if row_diff > 0:
        zero_padding = torch.zeros(row_diff, base_lm_head_weight.size(1), dtype=base_lm_head_weight.dtype, device=base_lm_head_weight.device)
        base_model.lm_head.weight = Parameter(torch.cat((base_lm_head_weight, zero_padding), dim=0))
    else:
        base_model.lm_head.weight = Parameter(base_lm_head_weight)

    return base_model


def main(
        base_model_path: str = "path/to/base-models",
        instruct_model_path: str = "path/to/instruct-models",
        finetuned_model_path: str = "path/to/fine-tuned-models",
        checkpoint_diff_output_path: str = "path/to/output-diff",
        is_peft: bool = False,
):
    """
    Args:
        base_model_path: Model name (on Hugging Face) of the base model or path to the base models directory.
        instruct_model_path: Model name (on Hugging Face) of the instruct model or path to the instruct models directory.
        finetuned_model_path: Model name (on Hugging Face) of the fine-tuned model or path to the fine-tuned models directory.
        is_peft: If True, the model is a PEFT model. Default is False.
    """

    print(" ===== Arguments ===== ")
    print(f"Base model path: {base_model_path}")
    print(f"Instruct model path: {instruct_model_path}")
    print(f"Fine-tuned model path: {finetuned_model_path}")
    print(f"Is PEFT: {is_peft}")
    print("========================\n\n")
        

    base_model = load_causal_lm_from_pretrained(base_model_path)
    instruct_model = load_causal_lm_from_pretrained(instruct_model_path)
    if is_peft:
        finetuned_model = load_causal_lm_from_pretrained(base_model_path)
        finetuned_model = PeftModel.from_pretrained(finetuned_model, finetuned_model_path)
        finetuned_model = finetuned_model.merge_and_unload() 
    else:
        finetuned_model = load_causal_lm_from_pretrained(finetuned_model_path) 

    base_model = update_embed_tokens_and_lm_head(base_model, instruct_model)
    finetuned_model = update_embed_tokens_and_lm_head(finetuned_model, instruct_model)
    
    base_weights = base_model.state_dict()
    instruct_weights = instruct_model.state_dict()
    finetuned_weights = finetuned_model.state_dict()

    assert finetuned_weights.keys() == base_weights.keys()
    assert instruct_weights.keys() == base_weights.keys()

    print("Applying diff (instruct - base) to finetuned model...")
    for key in tqdm(instruct_weights.keys(), desc="Transferring weights"):
        diff = instruct_weights[key].double() - base_weights[key].double()
        finetuned_weights[key] += diff

    finetuned_model.load_state_dict(finetuned_weights)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    finetuned_model.save_pretrained(checkpoint_diff_output_path)
    tokenizer.save_pretrained(checkpoint_diff_output_path)
    print(f"Saved diffed model to {checkpoint_diff_output_path}")

    del instruct_model, base_model, finetuned_model
    gc.collect()


if __name__ == "__main__":
    fire.Fire(main)