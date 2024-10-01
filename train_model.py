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
args: argparse.Namespace = parser.parse_args()


train_data: TrainData = TrainData.from_cbor_file(args.trainset)

def get_label_name_file(model_name: str) -> str:
    if model_name.endswith('.pth'):
        model_name = model_name[:-4]
    model_name += '_labels.json'
    return model_name

match args.arch:
    case 'v1':
        model = SignRecognizerV1(len(train_data.info.labels))
        tmp = model.trainModel(train_data)
        with open(get_label_name_file(tmp), 'w', encoding="utf-8") as f:
            json.dump(ModelInfoV1(train_data.info.labels, FRAME_SIZE).__dict__, f)

    case _:
        raise ValueError(f"Model architecture {args.arch} not found.")
