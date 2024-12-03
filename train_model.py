import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

from src.datasample import *
from src.model_class.sign_recognizer_v1 import *
from src.train_model.train import *
from src.train_model.CustomDataset import *
from src.train_model.parse_args import *
from src.train_model.init_train_data import *
from src.train_model.research_mode import *

import torch

args: Args = parse_args()

model: SignRecognizerV1 = None

if args.model_path:
    print("Loading model...", end="", flush=True)
    model = SignRecognizerV1.loadModelFromDir(args.model_path)
    print("[DONE]")

train_data, validation_data, model_info, weights_balance = init_train_set(args.trainset_path, args.validation_set_ratio, args.balance_weights, args.name)

if args.research:
    print("Starting research...")
    research_mode(args, model_info, train_data, validation_data, weights_balance, args.research_trial)
else:
    print("Starting training...")
    if model is None:
        model = SignRecognizerV1(model_info, device=args.device, dropout=args.dropout)
    validation_loss = train_model(model, args.device, train_data, validation_data, num_epochs=args.epoch, weights_balance=weights_balance)
    model.saveModel()
