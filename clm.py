from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from utils import get_argparser, print_args
from datasets import Dataset
import json
import os, torch, pandas as pd

def load_training_instances(code_path, code_col, doc_path):
    instances = list()
    code_df = pd.read_json(code_path, lines=True, orient="records", encoding="utf-8")
    code_df = code_df.sample(frac=1, random_state=42).reset_index(drop=True)
    instances.extend(code_df[code_col].tolist())

    if not doc_path:
        print("No documentation path provided, skipping documentation loading.")
        return instances

    for root, _, files in os.walk(doc_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    instances.append(content)
    return instances

def chunk_and_tokenize_batch(batch, tokenizer, max_length, hard_code_qwen=False):
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    if hard_code_qwen:
        eos = tokenizer.encode('<|endoftext|>') if tokenizer.eos_token_id is not None else []
    else:
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    if bos == eos:
        bos = []
    
    chunk_size = max_length - len(bos) - len(eos)

    input_ids = list()
    labels = list()

    for text in batch["content"]:
        is_fim = ("fim_prefix" in text) and ("fim_suffix" in text) and ("fim_middle" in text)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if is_fim and len(tokens) + len(bos) + len(eos) > max_length:
            print(f"Skipping FIM instance exceeding max length: {text[:30]}...")
            continue

        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i+chunk_size]
            chunk = bos + chunk + eos
            label_chunk = chunk.copy() 
            if len(chunk) < max_length:
                padding_length = max_length - len(chunk)
                chunk += [tokenizer.pad_token_id] * padding_length
                label_chunk += [-100] * padding_length

            input_ids.append(chunk)
            labels.append(label_chunk)

    return {"input_ids": input_ids, "labels": labels}

if __name__ == '__main__':  
    parser = get_argparser()
    args = parser.parse_args()
    print_args(args, title="PRE-TRAINING ARGUMENTS")

    model_name, max_length = args.model_name, args.max_length
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = max_length

    if torch.distributed.is_initialized():
        torch.distributed._set_static_graph()

    training_instances = load_training_instances(args.training_data, args.code_column, args.doc_path)
    dataset = Dataset.from_list([{"content": x} for x in training_instances]) 

    print(f"Dataset size: {len(dataset)}")
    print("First entry:\n", dataset[0])
    print("Last entry:\n", dataset[-1])

    tokenized_dataset = dataset.map(
        lambda batch: chunk_and_tokenize_batch(batch, tokenizer, max_length, hard_code_qwen=args.hard_code_qwen),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=args.num_proc
    )

    print(f"Tokenized dataset size: {len(tokenized_dataset)}")
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy=args.save_strategy,
        do_train=True,
        bf16=True,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.grad_acc_steps,
        save_total_limit=args.save_total_limit,
        ddp_find_unused_parameters=False,
        report_to=args.report_to,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    loss_history_path = os.path.join(args.out_dir, "loss_history.json")
    
    # Extract only the loss and learning rate logs
    log_history = [log for log in trainer.state.log_history if "loss" in log]
    
    with open(loss_history_path, "w") as f:
        json.dump(log_history, f, indent=4)
        
    print(f"Training loss history saved to {loss_history_path}")