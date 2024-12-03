import torch
from torch.utils.data import DataLoader
from src.train_model.train import CustomDataset
from src.model_class.sign_recognizer_v1 import ModelInfo

from src.datasample import *

def init_train_set(trainset_path: str, validation_ratio: float = 0.2, balance_weights: bool = True, model_name: str = None) -> tuple[DataLoader, DataLoader | None, ModelInfo, torch.Tensor | None]:
    """_summary_

    Args:
        trainset_path (str): _description_
        validation_ratio (float, optional): _description_. Defaults to 0.2.
        balance_weights (bool, optional): _description_. Defaults to True.
        model_name (str, optional): _description_. Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader | None, ModelInfo, torch.Tensor | None]: TrainData, ValidationData, ModelInfo, WeightsBalance
    """

    print("Loading trainset...", end="", flush=True)
    train_data: TrainData2 = TrainData2.from_cbor_file(trainset_path)
    print("[DONE]")

    model_info: ModelInfo = ModelInfo.build(
            train_data.info.memory_frame,
            train_data.info.active_gestures,
            train_data.info.labels,
            name=model_name,
            intermediate_layers=[])


    validation_dataloader: DataLoader = None
    if validation_ratio > 0:
        print("Splitting trainset...", end="", flush=True)
        train_data, validation_data = train_data.split_trainset(0.8)
        validation_dataloader: DataLoader = DataLoader(CustomDataset(validation_data.get_input_data(), validation_data.get_output_data(), model_info.layers[0]), batch_size=16, shuffle=True)
        print("[DONE]")
    train_dataloader: DataLoader = DataLoader(CustomDataset(train_data.get_input_data(), train_data.get_output_data(), model_info.layers[0]), batch_size=16, shuffle=True)


    balance_weights: bool = True if str(balance_weights).lower() in ["true", "1", "yes"] else False
    weigths_balance: torch.Tensor = None
    if balance_weights:
        weigths_balance = train_data.get_class_weights()

    return (train_dataloader, validation_dataloader, model_info, weigths_balance)
