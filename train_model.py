import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import shutil

from src.datasample import *
from src.model_class.transformer_sign_recognizer import *
from src.train_model.train import *
from src.train_model.CustomDataset import *
from src.train_model.parse_args import *
from src.train_model.init_train_data import *
from src.train_model.research_mode import *
from src.train_model.ConfusedSets import *
from src.train_model.TrainStat import *

import torch

args: Args = parse_args()

model: SignRecognizerTransformer = None

copy_previous_model: bool = False

if args.model_path:
    print("Loading model...", end="", flush=True)
    copy_previous_model = True
    model = SignRecognizerTransformer.loadModelFromDir(args.model_path, args.device)
    print("[DONE]")


current_time = time.strftime('%d-%m-%Y_%H-%M-%S')
train_data, validation_data, confuse_data, model_info, weights_balance, confused_sets, train_stats = init_train_set(args)

if model is None:
    model = SignRecognizerTransformer(model_info, args.d_model, args.num_heads, args.num_layers, device=args.device)

print("Starting training...")
train_stats = train_model(model, args.device, confused_sets, train_data, train_stats, validation_data, confuse_data, num_epochs=args.epoch, weights_balance=weights_balance)

nb_prev_model: int = 0
if copy_previous_model:
    path: str = args.model_path + "/previous_models/"
    os.makedirs(path, exist_ok=True)

    nb_prev_model = len(os.listdir(path))

    pth_file = model.info.name + ".pth"
    shutil.copy(args.model_path + "/" + pth_file, path)
    os.rename(path + pth_file, f"{path}/{model.info.name}_{nb_prev_model}.pth")

    model.saveModel(args.model_path)
    nb_prev_model =+ 1
else:
    args.model_path = model.saveModel()

train_stats.rename(model.info.name, nb_prev_model)
os.makedirs(args.model_path + "/train_stats/", exist_ok=True)
train_stats.save(f"{args.model_path}/train_stats/{train_stats.name}.json")
