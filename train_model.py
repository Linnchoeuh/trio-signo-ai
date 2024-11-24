import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

from src.datasample import *
from src.model_class.sign_recognizer_v1 import *

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


args: argparse.Namespace = parser.parse_args()

train_data: TrainData2 = TrainData2.from_cbor_file(args.trainset)

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


match args.arch:
    case 'v1':
        model = SignRecognizerV1(ModelInfo.build(
            train_data.info.memory_frame,
            train_data.info.active_gestures,
            train_data.info.labels,
            name=args.name,
            intermediate_layers=[8]), device=device)
        validation_data: TrainData2 = None
        train_data, validation_data = train_data.split_trainset(0.8)
        # print(train_data.getNumberOfSamples(), validation_data.getNumberOfSamples())

        model.trainModel(train_data, num_epochs=int(args.epoch), validation_data=validation_data)
        model.saveModel()

    case _:
        raise ValueError(f"Model architecture {args.arch} not found.")
