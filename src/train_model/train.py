import time

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

from src.train_model.AccuracyCalculator import AccuracyCalculator
from src.train_model.CustomDataset import CustomDataset
from src.model_class.transformer_sign_recognizer import *

def train_epoch_run_model(model: SignRecognizerTransformer, criterion: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Args:
        model (SignRecognizerTransformer): _description_
        criterion (nn.CrossEntropyLoss): _description_
        inputs (torch.Tensor): _description_
        labels (torch.Tensor): _description_

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple(loss, outputs)
    """
    outputs: torch.Tensor = model(inputs)
    return (criterion(outputs, labels), outputs)

def train_epoch_optimize(optimizer: optim.Optimizer, loss: torch.Tensor):
    """_summary_

    Args:
        optimizer (optim.Optimizer): _description_
        loss (torch.Tensor): _description_
    """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_epoch(model: SignRecognizerTransformer, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> tuple[torch.Tensor, AccuracyCalculator]:
    """Will run the model and then optimize it.

    Args:
        model (SignRecognizerTransformer): Model to run
        dataloader (DataLoader): Data to run the model on
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer

    Returns:
        tuple[torch.Tensor, AccuracyCalculator]: tuple(loss, accuracy_calculator)
    """
    model.train()
    accuracy_calculator: AccuracyCalculator = AccuracyCalculator(model.info.labels)

    for inputs, labels in dataloader:
        # inputs, labels = inputs.to(model.device), labels.to(model.device)
        loss, outputs = train_epoch_run_model(model, criterion, inputs, labels)
        train_epoch_optimize(optimizer, loss)

        accuracy_calculator.calculate_accuracy(outputs, labels)

    return loss, accuracy_calculator

def validation_epoch(model: SignRecognizerTransformer, dataloader: DataLoader, criterion: nn.Module) -> tuple[torch.Tensor, AccuracyCalculator]:
    """Will run the model without doing the optimization part (Use <strong>train_epoch</strong> function for that).

    Args:
        model (SignRecognizerV1): Model to run
        dataloader (DataLoader): Data to run the model on
        criterion (nn.Module): Loss function

    Returns:
        tuple[torch.Tensor, AccuracyCalculator]: tuple(loss, accuracy_calculator)
    """
    model.eval()
    accuracy_calculator: AccuracyCalculator = AccuracyCalculator(model.info.labels)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            loss, outputs = train_epoch_run_model(model, criterion, inputs, labels)

            accuracy_calculator.calculate_accuracy(outputs, labels)

    return loss, accuracy_calculator

def log_validation_info(val_acc: AccuracyCalculator, loss: torch.Tensor, val_loss: torch.Tensor):
    loss_diff: float = abs(loss.item() - val_loss.item())
    mean_loss: float = (loss + val_loss.item()) / 2

    validation_avg_acc, _ = val_acc.get_accuracy()
    print(f"\tValidation Loss: {val_loss.item():.4f}, " +
        f"Validation accuracy: {(validation_avg_acc * 100):.2f}%")
    val_acc.print_accuracy_table()
    print(f"\tLoss Diff: {loss_diff:.4f}, Mean Loss: {mean_loss:.4f}")

def train_model(model: SignRecognizerTransformer, device: torch.device,
                train_data: DataLoader, validation_data: DataLoader = None,
                num_epochs: int = 20, weights_balance: torch.Tensor = None,
                learning_rate: float = 0.001, validation_interval: int = 2, silent: bool = False) -> float:

    model.to(device)

    criterion: nn.Module = nn.CrossEntropyLoss()
    if weights_balance is not None:
        criterion.weight = weights_balance
    criterion.to(device)
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    start_time = time.time()
    remain_time: str = "Estimating..."
    final_loss: float = 0
    validation_was_runned: bool = False
    for epoch in range(num_epochs):
        validation_was_runned = False
        loss, train_acc = train_epoch(model, train_data, criterion, optimizer)
        final_loss = loss.item()

        if not silent:
            print(f"--- " +
                f"Epoch [{epoch+1}/{num_epochs}], " +
                f"Remaining time: {remain_time}, " +
                f"Learning Rate: {(optimizer.param_groups[0]['lr']):.6f}" +
                f" ---")
            train_avg_acc, _ = train_acc.get_accuracy()
            print(f"\tTrain Loss: {loss.item():.4f}, " +
                f"Train Accuracy: {(train_avg_acc * 100):.2f}%")
            train_acc.print_accuracy_table()



        if validation_data is not None and validation_interval > 1 and epoch % validation_interval == validation_interval - 1:
            validation_was_runned = True
            val_loss, val_acc = validation_epoch(model, validation_data, criterion)
            final_loss = val_loss.item()

            if not silent:
                log_validation_info(val_acc, loss, val_loss)



        scheduler.step(final_loss)
        remain_time = time.strftime('%H:%M:%S', time.gmtime(((time.time() - start_time) / (epoch+1)) * (num_epochs - epoch - 1)))

    if validation_data is not None and not validation_was_runned:
        val_loss, val_acc = validation_epoch(model, validation_data, criterion)
        final_loss = val_loss.item()

        if not silent:
            log_validation_info(val_acc, loss, val_loss)


    return final_loss
