import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import optuna

from src.datasample import *
from src.model_class.sign_recognizer_v1 import *
from src.train_model.train import *
from src.train_model.CustomDataset import *

import torch

parser: argparse.ArgumentParser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--trainset',
    help='File path to the training set.',
    required=True)
parser.add_argument(
    '--arch',
    help='Model architecture to use. (Available: v1)',
    required=False,
    default='v1')
parser.add_argument(
    '--memory-frame',
    help='Number of frame in the past the model will see (Default: None (Maximum possible frame in the past the trainset have))',
    required=False,
    default=None)
parser.add_argument(
    '--name',
    help='Name of the model',
    required=False,
    default=None)
parser.add_argument(
    '--epoch',
    help='Name of the model',
    required=False,
    default=20)
parser.add_argument(
    '--device',
    help='The device to use for training options: (cpu, gpu, cuda, mps)',
    required=False,
    default="gpu")
parser.add_argument(
    '--balance-weights',
    help='Balance the weight for the loss function so no label is overrepresented.',
    required=False,
    default=True)
parser.add_argument(
    '--neuron-max',
    help='Maximum number of neuron per layer.',
    required=False,
    default=256)
parser.add_argument(
    '--neuron-min',
    help='Minimum number of neuron per layer.',
    required=False,
    default=16)
parser.add_argument(
    '--layer-max',
    help='Maximum number of layer.',
    required=False,
    default=3)
parser.add_argument(
    '--layer-min',
    help='Minimum number of layer.',
    required=False,
    default=1)
parser.add_argument(
    '--validation-set-ratio',
    help='Ratio of the trainset that will be used for the validation set.',
    required=False,
    default=0.2)
parser.add_argument(
    '--min-dropout',
    help='Dropout value for the model.',
    required=False,
    default=0.3)
parser.add_argument(
    '--max-dropout',
    help='Dropout value for the model.',
    required=False,
    default=0.5)

args: argparse.Namespace = parser.parse_args()

max_neuron: int = int(args.neuron_max)
min_neuron: int = int(args.neuron_min)
max_layer: int = int(args.layer_max)
min_layer: int = int(args.layer_min)
max_dropout: float = float(args.max_dropout)
min_dropout: float = float(args.min_dropout)

print("Loading trainset...", end="")
train_data: TrainData2 = TrainData2.from_cbor_file(args.trainset)
print("[DONE]")
model_info: ModelInfo = ModelInfo.build(
            train_data.info.memory_frame,
            train_data.info.active_gestures,
            train_data.info.labels,
            name=args.name,
            intermediate_layers=[])

validation_ratio: float = float(args.validation_set_ratio)
validation_dataloader: DataLoader = None
if validation_ratio > 0:
    print("Splitting trainset...", end="")
    train_data, validation_data = train_data.split_trainset(0.8)
    validation_dataloader: DataLoader = DataLoader(CustomDataset(validation_data.get_input_data(), validation_data.get_output_data(), model_info.layers[0]), batch_size=16, shuffle=True)
    print("[DONE]")
train_dataloader: DataLoader = DataLoader(CustomDataset(train_data.get_input_data(), train_data.get_output_data(), model_info.layers[0]), batch_size=16, shuffle=True)


balance_weights: bool = True if str(args.balance_weights).lower() in ["true", "1", "yes"] else False
weigths_balance: torch.Tensor = None
if balance_weights:
    weigths_balance = train_data.get_class_weights()

print("Device selection... ", end="")
device = torch.device("cpu")
if args.device in ["gpu", "cuda"] and torch.cuda.is_available():
    # Check for CUDA (NVIDIA GPU)
    device = torch.device("cuda")
    print("Using NVIDIA GPU with CUDA")
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()), "ID:", torch.cuda.current_device())
elif args.device in ["gpu", "mps"] and torch.backends.mps.is_available():  # On ROCm-enabled PyTorch builds
    # Check for ROCm (AMD GPU)
    device = torch.device("mps")
    print("Using AMD GPU with ROCm")
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()), "ID:", torch.cuda.current_device())
else:
    # Default to CPU
    print("Using CPU")

def objective(trial: optuna.trial.Trial) -> float:
    num_layers = trial.suggest_int("num_layers", min_layer, max_layer)
    layers: list[int] = []
    for i in range(num_layers):
        layers.append(trial.suggest_int(f"hidden_size_layer_{i}", min_neuron, max_neuron))
    dropout = trial.suggest_float("dropout", min_dropout, max_dropout)

    model_info.set_intermediate_layers(layers)

    validation_loss = 0

    match args.arch:
        case 'v1':
            model = SignRecognizerV1(model_info, device=device, dropout=dropout)

            validation_loss = train_model(model, device, train_dataloader, validation_dataloader, num_epochs=5, weights_balance=weigths_balance, validation_interval=-1)

            # model.saveModel()


        case _:
            raise ValueError(f"Model architecture {args.arch} not found.")

    return validation_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

sorted_trials = sorted(study.trials, key=lambda t: t.value)

# Display the ranking
print("Ranked Trials:")
for rank, trial in enumerate(sorted_trials, 1):
    print(f"Rank {rank}: Value={trial.value:.4f}, Params={trial.params}")
