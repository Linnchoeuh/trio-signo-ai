import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

from src.datasample import *
from src.model_class.sign_recognizer_v1 import *

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
    help='Number of frame in the past the model will see (Default: None (Maximum possible frame in the past the trainset have))',
    required=False,
    default=None)
parser.add_argument(
    '--name',
    help='Name of the model',
    required=False,
    default=None)

args: argparse.Namespace = parser.parse_args()

train_data: TrainData2 = TrainData2.from_cbor_file(args.trainset)

memory_frame: int = train_data.info.memory_frame
if args.memory_frame is not None and int(args.memory_frame) <= memory_frame:
    memory_frame = int(args.memory_frame)



match args.arch:
    case 'v1':
        model = SignRecognizerV1(ModelInfo.build(
            memory_frame,
            train_data.info.active_gestures,
            train_data.info.labels,
            name=args.name,
            intermediate_layers=[64, 64]))
        validation_data: TrainData2 = None
        train_data, validation_data = train_data.split_trainset(0.8)
        # print(train_data.getNumberOfSamples(), validation_data.getNumberOfSamples())

        model.trainModel(train_data, validation_data=validation_data)
        model.saveModel()

    case _:
        raise ValueError(f"Model architecture {args.arch} not found.")
