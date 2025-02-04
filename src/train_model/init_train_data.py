import torch
from torch.utils.data import DataLoader
from src.train_model.train import CustomDataset
from src.model_class.transformer_sign_recognizer import ModelInfo, SignRecognizerTransformerDataset

from src.datasample import *
from src.datasamples import *

def init_train_set(trainset_path: str, validation_ratio: float = 0.2, balance_weights: bool = True, batch_size: int = 16, model_name: str = None, device: torch.device = torch.device("cpu")) -> tuple[DataLoader, DataLoader | None, ModelInfo, torch.Tensor | None]:
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
    train_data: DataSamples = DataSamples.fromCborFile(trainset_path)
    print("[DONE]")

    model_info: ModelInfo = ModelInfo.build(
            info=train_data.info,
            name=model_name)


    train_in_data, train_out_data, validation_in_data, validation_out_data = train_data.toTensors(device, validation_ratio)


    validation_dataloader: DataLoader = None
    if validation_ratio > 0:
        validation_dataloader: DataLoader = DataLoader(SignRecognizerTransformerDataset(validation_in_data, validation_out_data), batch_size=batch_size, shuffle=True)
    train_dataloader: DataLoader = DataLoader(SignRecognizerTransformerDataset(train_in_data, train_out_data), batch_size=batch_size, shuffle=True)


    balance_weights: bool = True if str(balance_weights).lower() in ["true", "1", "yes"] else False
    weigths_balance: torch.Tensor = None
    if balance_weights:
        weigths_balance = train_data.getClassWeights()

    return (train_dataloader, validation_dataloader, model_info, weigths_balance)
