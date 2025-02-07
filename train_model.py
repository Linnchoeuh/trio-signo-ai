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

import torch

args: Args = parse_args()

model: SignRecognizerTransformer = None

if args.model_path:
    print("Loading model...", end="", flush=True)
    model = SignRecognizerTransformer.loadModelFromDir(args.model_path)
    print("[DONE]")


train_data, validation_data, model_info, weights_balance = init_train_set(args.trainset_path, args.validation_set_ratio, args.balance_weights, 32, args.name)

# if args.research:
#     print("Starting research...")
#     research_mode(args, model_info, train_data, validation_data, weights_balance, args.research_trial)
# else:

if model is None:
    model = SignRecognizerTransformer(model_info, args.d_model, args.num_heads, args.num_layers, device=args.device)
print("Starting training...")
validation_loss = train_model(model, args.device, train_data, validation_data, num_epochs=args.epoch, weights_balance=weights_balance)
model.saveModel()
