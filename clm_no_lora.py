from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset

import os, torch, argparse, pandas as pd


def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """

    parser = argparse.ArgumentParser(description='optional arguments')
    parser.add_argument('--max_model_length', '-l',
                        metavar='INT',
                        dest='max_length',
                        required=False,
                        type=int,
                        default=2048,
                        help='Maximum length of the model (default: 2048)')
    parser.add_argument('--output_dir', '-o',
                        metavar='PATH',
                        dest='out_dir',
                        required=False,
                        type=str,
                        default='output',
                        help='Name of the directory where to save the model')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--model_name', '-m',
                        metavar='NAME',
                        dest='model_name',
                        required=True,
                        type=str,
                        help='Name of the model to use for training (e.g., Qwen/Qwen2.5-Coder-7B)')
    required.add_argument('--training_data', '-t',
                        metavar='PATH',
                        dest='training_data',
                        required=True,
                        type=str,
                        help='Path to the training data file')
    required.add_argument('--batch_size', '-bs',
                        dest='batch_size',
                        required=True,
                        type=int,
                        help='Number of samples to process in one training step')
    required.add_argument('--epochs', '-e',
                        dest='epochs',
                        required=True,
                        type=int,
                        help='Number of epochs to train the model for')
    required.add_argument('--learning_rate', '-lr',
                        dest='learning_rate',
                        type=float,
                        required=True,
                        help='Value of the learning rate to use for training the model')
    required.add_argument('--lr_scheduler_type', '-lrt',
                        dest='lr_scheduler_type',
                        required=True,
                        type=str,
                        help='Scheduler type to use for learning rate adjustment (e.g., linear, cosine)')
    required.add_argument('--gradient_checkpointing', '-gc',
                        dest='gradient_checkpointing',
                        action='store_true',
                        help='Enable gradient checkpointing')
    required.add_argument('--gradient_accumulation_steps', '-gas',
                        dest='gradient_accumulation_steps',
                        required=True,
                        type=int,
                        help='Number of steps to accumulate gradients before performing an optimization step')
    required.add_argument('--logging_steps', '-lgs',
                        dest='logging_steps',
                        required=True,
                        type=int,
                        help='Number of steps between logging training progress')
    required.add_argument('--seed', '-s',
                        dest='seed',
                        required=True,
                        type=int,
                        help='Random seed for reproducibility')
    return parser


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
        tokens = tokenizer.encode(text, add_special_tokens=False)

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

    # Print all the arguments
    print("=" * 40)
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=" * 40)

    model_name = args.model_name
    max_length = args.max_length

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = max_length

    # if torch.distributed.is_initialized():
    #     torch.distributed._set_static_graph()
    
    training_instances = list()
    training_code_df = pd.read_json(args.training_data, lines=True, orient="records", encoding="utf-8")
    training_instances.extend(training_code_df["text"].tolist())

    dataset = Dataset.from_list([{"content": x} for x in training_instances])
    dataset = dataset.shuffle(seed=42)

    print(f"Dataset size: {len(dataset)}")
    print("First entry:\n", dataset[0])
    print("Last entry:\n", dataset[-1])

    is_hard_code_qwen = "Qwen" in model_name

    tokenized_dataset = dataset.map(
        lambda batch: chunk_and_tokenize_batch(batch, tokenizer, max_length, is_hard_code_qwen),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=32
    )

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        do_train=True,
        bf16=True,
        fp16=False,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=100,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=args.seed
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    trainer.train()