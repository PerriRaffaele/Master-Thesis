import argparse

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser with grouped arguments
    """
    parser = argparse.ArgumentParser(description='Arguments for pre-training a model with Causal Language Modeling (CLM) using LoRA.')

    # Model arguments
    model_group = parser.add_argument_group('Model arguments')
    model_group.add_argument('--model_name', '-m',
                            metavar='NAME',
                            dest='model_name',
                            required=True,
                            type=str,
                            help='Name of the model to use for training (e.g., Qwen/Qwen2.5-Coder-7B)')
    model_group.add_argument('--adapter_path', '-a',
                            metavar='PATH',
                            dest='adapter_path',
                            required=False,
                            type=str,
                            default=None,
                            help='Path to a pre-trained LoRA adapter to load (default: None)')
    model_group.add_argument('--adapter_revision', '-ar',
                            metavar='STR',
                            dest='adapter_revision',
                            required=False,
                            type=str,
                            default='main',
                            help='Revision of the adapter to use (default: main)')
    model_group.add_argument('--max_model_length', '-l',
                            metavar='INT',
                            dest='max_length',
                            required=False,
                            type=int,
                            default=2048,
                            help='Maximum length of the model (default: 2048)')
    model_group.add_argument('--hard_code_qwen',
                             action='store_true',
                             dest='hard_code_qwen',
                             default=False,
                             help='Whether to hard-code Qwen tokenizer special tokens handling (default: False)')

    # LoRA arguments
    lora_group = parser.add_argument_group('LoRA arguments')
    lora_group.add_argument('--lora_r',
                            metavar='INT',
                            dest='lora_r',
                            required=False,
                            type=int,
                            default=16,
                            help='LoRA rank (default: 16)')
    lora_group.add_argument('--lora_alpha',
                            metavar='INT',
                            dest='lora_alpha',
                            required=False,
                            type=int,
                            default=32,
                            help='LoRA alpha (default: 32)')
    lora_group.add_argument('--lora_dropout',
                            metavar='FLOAT',
                            dest='lora_dropout',
                            required=False,
                            type=float,
                            default=0.05,
                            help='LoRA dropout (default: 0.05)')
    lora_group.add_argument('--lora_target_modules',
                            metavar='STR',
                            dest='lora_target_modules',
                            nargs='+',
                            required=False,
                            default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                            help='LoRA target modules (default: common transformer modules)')

    # Training arguments
    train_group = parser.add_argument_group('Training arguments')
    train_group.add_argument('--output_dir', '-o',
                            metavar='PATH',
                            dest='out_dir',
                            required=False,
                            type=str,
                            default='output',
                            help='Directory to save the model')
    train_group.add_argument('--batch_size', '-b',
                            metavar='INT',
                            dest='batch_size',
                            required=False,
                            type=int,
                            default=1,
                            help='Batch size per device (default: 1)')
    train_group.add_argument('--epochs', '-e',
                            metavar='INT',
                            dest='epochs',
                            required=False,
                            type=int,
                            default=5,
                            help='Number of training epochs (default: 5)')
    train_group.add_argument('--gradient_accumulation_steps', '-g',
                            metavar='INT',
                            dest='grad_acc_steps',
                            required=False,
                            type=int,
                            default=4,
                            help='Gradient accumulation steps (default: 4)')
    train_group.add_argument('--learning_rate', '-lr',
                            metavar='FLOAT',
                            dest='learning_rate',
                            required=False,
                            type=float,
                            default=5e-5,
                            help='Learning rate (default: 5e-5)')
    train_group.add_argument('--weight_decay',
                            metavar='FLOAT',
                            dest='weight_decay',
                            required=False,
                            type=float,
                            default=0.01,
                            help='Weight decay for optimizer (default: 0.01)')
    train_group.add_argument('--warmup_ratio',
                            metavar='FLOAT',
                            dest='warmup_ratio',
                            required=False,
                            type=float,
                            default=0.1,
                            help='Warmup ratio for learning rate scheduler (default: 0.1)')
    train_group.add_argument('--logging_steps',
                            metavar='INT',
                            dest='logging_steps',
                            required=False,
                            type=int,
                            default=10,
                            help='Logging steps (default: 10)')
    train_group.add_argument('--save_strategy',
                            metavar='STR',
                            dest='save_strategy',
                            required=False,
                            type=str,
                            default='epoch',
                            help='Save strategy (default: epoch)')
    train_group.add_argument('--save_total_limit',
                            metavar='INT',
                            dest='save_total_limit',
                            required=False,
                            type=int,
                            default=100,
                            help='Max number of checkpoints to keep (default: 100)')
    train_group.add_argument('--report_to',
                            metavar='STR',
                            dest='report_to',
                            required=False,
                            type=str,
                            default='none',
                            help='Reporting tool (default: none)')

    # Data arguments
    data_group = parser.add_argument_group('Data arguments')
    data_group.add_argument('--training_data', '-t',
                            metavar='PATH',
                            dest='training_data',
                            required=True,
                            type=str,
                            help='Path to the training data file')
    data_group.add_argument('--code_column', '-c',
                            metavar='STR',
                            dest='code_column',
                            required=False,
                            type=str,
                            default='code',
                            help='Name of the column containing the code (default: code)')
    data_group.add_argument('--prompt_column', '-p',
                            metavar='STR',
                            dest='prompt_column',
                            required=False,
                            type=str,
                            default='prompt',
                            help='Name of the column containing the prompt (default: prompt)')
    data_group.add_argument('--completion_column', '-r',
                            metavar='STR',
                            dest='completion_column',
                            required=False,
                            type=str,
                            default='completion',
                            help='Name of the column containing the completion/response (default: completion)')
    data_group.add_argument('--documentation_path', '-d',
                            metavar='PATH',
                            dest='doc_path',
                            required=False,
                            default=None,
                            type=str,
                            help='Path to the documentation directory')
    data_group.add_argument('--num_proc',
                            metavar='INT',
                            dest='num_proc',
                            required=False,
                            type=int,
                            default=32,
                            help='Number of processes for tokenization (default: 32)')
    

    return parser


def print_args(args, title="Arguments"):
    """
    Print the arguments in a formatted way
    """
    print("#" * 40)
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("#" * 40)