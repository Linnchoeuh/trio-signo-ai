import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

from src.datasample import *
from src.model_class.alphabet_recognizer_v2 import SignRecognizerV2

train_data: TrainData = TrainData.from_json_file("trainset_23-09-2024_03-22-55.td.json")

print(train_data.get_input_data()[:2])

model = SignRecognizerV2(len(train_data.info.labels))

model.train(train_data)
