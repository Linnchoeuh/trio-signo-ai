import argparse
from dataclasses import dataclass, field

import torch

@dataclass
class Args:
    trainset_path: str = None
    # arch: str = "v1"
    memory_frame: int = None
    name: str = None
    epoch: int = 20
    device_type: str = "gpu"
    device: torch.device = None
    balance_weights: bool = True
    # min_neuron: int = 16
    # max_neuron: int = 256
    # min_layer: int = 1
    # max_layer: int = 3
    validation_set_ratio: float = 0.2
    # min_dropout: float = 0.3
    # max_dropout: float = 0.5
    dropout: float = 0.3
    # research: bool = False
    # research_trial: int = 50
    model_path: str = None
    d_model: int = 32
    num_heads: int = 8
    num_layers: int = 3
    ff_dim: int = None
    confusing_label: dict[str, str] = field(default_factory=dict)
    batch_size: int = 32


def parse_args() -> Args:
    args = Args()

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--trainset',
        help='File path to the training set.',
        required=True,
        type=str)
    # parser.add_argument(
    #     '--arch',
    #     help='(Unused at the moment) Model architecture to use. (Available: v1)',
    #     required=False,
    #     default=args.arch,
    #     type=str)
    parser.add_argument(
        '--memory-frame',
        help='Number of frame in the past the model will see (Default: None (Maximum possible frame in the past the trainset have))',
        required=False,
        default=args.memory_frame,
        type=int)
    parser.add_argument(
        '--name',
        help='Name of the model',
        required=False,
        default=args.name,
        type=str)
    parser.add_argument(
        '--epoch',
        help='Number of training iteration',
        required=False,
        default=args.epoch,
        type=int)
    parser.add_argument(
        '--device',
        help='The device to use for training options: (cpu, gpu, cuda, mps)',
        required=False,
        default=args.device_type,
        type=str)
    parser.add_argument(
        '--balance-weights',
        help='Balance the weight for the loss function so no label is overrepresented.',
        required=False,
        action='store_true')
    # parser.add_argument(
    #     '--min-neuron',
    #     help='(Only with research mode) Minimum number of neuron per layer.',
    #     required=False,
    #     default=args.min_neuron,
    #     type=int)
    # parser.add_argument(
    #     '--max-neuron',
    #     help='(Only with research mode) Maximum number of neuron per layer.',
    #     required=False,
    #     default=args.max_neuron,
    #     type=int)
    # parser.add_argument(
    #     '--min-layer',
    #     help='(Only with research mode) Minimum number of layer.',
    #     required=False,
    #     default=args.min_layer,
    #     type=int)
    # parser.add_argument(
    #     '--max-layer',
    #     help='(Only with research mode) Maximum number of layer.',
    #     required=False,
    #     default=args.max_layer,
    #     type=int)
    parser.add_argument(
        '--validation-set-ratio',
        help='Ratio of the trainset that will be used for the validation set.',
        required=False,
        default=args.validation_set_ratio,
        type=float)
    parser.add_argument(
        '--dropout',
        help='Dropout value for the model.',
        required=False,
        default=args.dropout,
        type=float)
    # parser.add_argument(
    #     '--min-dropout',
    #     help='(Only with research mode) Minimum dropout value for the model.',
    #     required=False,
    #     default=args.min_dropout,
    #     type=float)
    # parser.add_argument(
    #     '--max-dropout',
    #     help='(Only with research mode) Maximum dropout value for the model.',
    #     required=False,
    #     default=args.max_dropout,
    #     type=float)
    # parser.add_argument(
    #     '--research',
    #     help='Change the training mode to research mode.',
    #     required=False,
    #     default=args.research,
    #     action='store_true')
    # parser.add_argument(
    #     '--research-trial',
    #     help='Number of trial',
    #     required=args.model_path,
    #     default=args.research_trial,
    #     type=int)
    parser.add_argument(
        '--model',
        help='Path to the model. (Must be a folder containing a .json and a .pth file)',
        required=args.model_path,
        default=None,
        type=str)
    parser.add_argument(
        '--d-model',
        help='Dimension of the model embedding.',
        required=False,
        default=args.d_model,
        type=int)
    parser.add_argument(
        '--num-heads',
        help='Number of attention head in the model.',
        required=False,
        default=args.num_heads,
        type=int)
    parser.add_argument(
        '--num-layers',
        help='Number of layers in the model.',
        required=False,
        default=args.num_layers,
        type=int)
    parser.add_argument(
        '--ff-dim',
        help='Dimension of the feed-forward network in the model.',
        required=False,
        default=args.ff_dim,
        type=int)
    parser.add_argument(
        '--confusing-label', "-c",
        help='Give a pair of label that are ambiguous to the model.',
        required=False,
        default=None,
        type=str,
        action='extend',
        nargs=2)
    parser.add_argument(
        '--batch-size',
        help='Batch size for the training. (Bigger batch size will use more memory but will train faster)',
        required=False,
        default=32,
        type=int)

    term_args: argparse.Namespace = parser.parse_args()

    args.trainset_path = term_args.trainset
    # args.arch = term_args.arch
    args.memory_frame = term_args.memory_frame
    args.name = term_args.name
    args.epoch = term_args.epoch

    print("Confusing label registered:")
    i: int = 0
    while i < len(term_args.confusing_label):
        c_label = term_args.confusing_label[i]
        c_label2 = term_args.confusing_label[i+1]
        try:
            assert args.confusing_label.get(c_label) is None, f"Label \"{c_label}\" already in the list. If the \"{c_label}\" is responsible of more than one label, do something like this:\n-c \"{c_label2}\" \"{c_label}\"\nInstead of:\n-c \"{c_label}\" \"{c_label2}\""
        except AssertionError as e:
            print("AssertionError:", e)
            exit(1)
        args.confusing_label[c_label] = c_label2
        print(f"\t{c_label} <=> {c_label2}")
        i += 2

    args.device_type = term_args.device
    print("Requested device:", args.device_type)
    print("CUDA available:", torch.cuda.is_available())
    print("ROCm available:", torch.backends.mps.is_available())
    print("Device selection... ", end="", flush=True)
    args.device = torch.device("cpu")
    if args.device_type in ["gpu", "cuda"] and torch.cuda.is_available():
        # Check for CUDA (NVIDIA GPU)
        args.device = torch.device("cuda")
        print("Using NVIDIA GPU with CUDA")
        print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()), "ID:", torch.cuda.current_device())
    elif args.device_type in ["gpu", "mps"] and torch.backends.mps.is_available():  # On ROCm-enabled PyTorch builds
        # Check for ROCm (AMD GPU)
        args.device = torch.device("mps")
        print("Using AMD GPU with ROCm")
        print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()), "ID:", torch.cuda.current_device())
    else:
        # Default to CPU
        print("Using CPU")

    args.balance_weights = term_args.balance_weights
    # args.min_neuron = term_args.min_neuron
    # args.max_neuron = term_args.max_neuron
    # args.min_layer = term_args.min_layer
    # args.max_layer = term_args.max_layer
    args.validation_set_ratio = float(term_args.validation_set_ratio)
    # args.min_dropout = term_args.min_dropout
    # args.max_dropout = term_args.max_dropout
    args.dropout = float(term_args.dropout)
    # args.research = term_args.research
    # args.research_trial = term_args.research_trial
    args.model_path = term_args.model
    args.d_model = int(term_args.d_model)
    args.num_heads = int(term_args.num_heads)
    args.num_layers = int(term_args.num_layers)
    if term_args.ff_dim is not None:
        args.ff_dim = int(term_args.ff_dim)
    args.batch_size = int(term_args.batch_size)
    return args
