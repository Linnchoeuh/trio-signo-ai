import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

from src.datasample import *
from src.model_class.sign_recognizer_v1 import *
from src.model_class.sign_recognizer_v2 import *

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
    '--memory_frame',
    help='Number of frame in the past the model will see',
    required=False,
    default=None)
parser.add_argument(
    '--name',
    help='Name of the model',
    required=False,
    default=None)

args: argparse.Namespace = parser.parse_args()

train_data: TrainData = TrainData.from_cbor_file(args.trainset)

memory_frame: int = args.memory_frame
if memory_frame is None:
    memory_frame = train_data.info.memory_frame
else:
    memory_frame = int(memory_frame)



match args.arch:
    case 'v1':
        model = SignRecognizerV1(len(train_data.info.labels), memory_frame)
        model.trainModel(train_data)
        model.saveModel(train_data, args.name)

    case _:
        raise ValueError(f"Model architecture {args.arch} not found.")
