import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

from src.datasample import *
from src.model_class.transformer_sign_recognizer import *
from src.train_model.train import *
from src.train_model.CustomDataset import *
from src.train_model.parse_args import *
from src.train_model.init_train_data import *
from src.train_model.research_mode import *
from src.train_model.ConfusedSets import *

import torch

args: Args = parse_args()

model: SignRecognizerTransformer = None

if args.model_path:
    print("Loading model...", end="", flush=True)
    model = SignRecognizerTransformer.loadModelFromDir(args.model_path)
    print("[DONE]")

train_data, validation_data, confuse_data, model_info, weights_balance, confused_sets = init_train_set(args)

if model is None:
    model = SignRecognizerTransformer(model_info, args.d_model, args.num_heads, args.num_layers, device=args.device)
print("Starting training...")
validation_loss = train_model(model, args.device, confused_sets, train_data, validation_data, confuse_data, num_epochs=args.epoch, weights_balance=weights_balance)
model.saveModel()
