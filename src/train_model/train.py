import time

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

from src.train_model.AccuracyCalculator import AccuracyCalculator
from src.train_model.CustomDataset import CustomDataset
from src.model_class.transformer_sign_recognizer import *
from src.train_model.ConfusedSets import *
from src.train_model.TrainStat import *


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

def cross_entropy_train_epoch(model: SignRecognizerTransformer, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> tuple[torch.Tensor, AccuracyCalculator]:
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
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        loss, outputs = train_epoch_run_model(model, criterion, inputs, labels)
        # print(loss.shape)
        train_epoch_optimize(optimizer, loss)

        accuracy_calculator.calculate_accuracy(outputs, labels)

    return loss, accuracy_calculator

def triplet_margin_train_epoch(model: SignRecognizerTransformer, dataloader: DataLoader, criterion: nn.TripletMarginLoss, optimizer: optim.Optimizer, confused_sets: ConfusedSets) -> tuple[torch.Tensor]:
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

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs: torch.Tensor = model(inputs, return_embeddings=True)
        positive_samples: torch.Tensor = model(confused_sets.getPositiveSamples(labels).to(model.device), return_embeddings=True)
        negative_samples: torch.Tensor = model(confused_sets.getNegativeSamples(labels).to(model.device), return_embeddings=True)
        # print(outputs.shape, positive_samples.shape, negative_samples.shape, "\n", outputs, "\n", positive_samples, "\n", negative_samples)
        loss: torch.Tensor = criterion(outputs, positive_samples, negative_samples)
        train_epoch_optimize(optimizer, loss)


    return loss

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

def train_model(model: SignRecognizerTransformer, device: torch.device, confused_sets: ConfusedSets,
                train_data: DataLoader, train_stats: TrainStat, validation_data: DataLoader = None, confuse_data: DataLoader = None,
                num_epochs: int = 20, weights_balance: torch.Tensor = None,
                learning_rate: float = 0.001, validation_interval: int = 2, silent: bool = False) -> TrainStat:

    model.to(device)

    cross_entropy_criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    triplet_margin_criterion: nn.TripletMarginLoss = nn.TripletMarginLoss(margin=1.0)
    if weights_balance is not None:
        cross_entropy_criterion.weight = weights_balance
    cross_entropy_criterion.to(device)
    triplet_margin_criterion.to(device)

    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    total_time: float = time.time()
    train_epoch_durations: list[float] = []
    validation_epoch_durations: list[float] = []


    remain_time: str = "Estimating..."
    final_loss: float = 0
    validation_was_runned: bool = False
    for epoch in range(num_epochs):
        validation_was_runned = False
        total_loss: torch.Tensor = None

        start_time: float = time.time()
        ce_loss, train_acc = cross_entropy_train_epoch(model, train_data, cross_entropy_criterion, optimizer)
        total_loss = ce_loss
        if confuse_data is not None:
            tm_loss = triplet_margin_train_epoch(model, confuse_data, triplet_margin_criterion, optimizer, confused_sets)
            # total_loss = torch.cat((ce_loss, tm_loss))
            total_loss += tm_loss
        final_loss: torch.types.Number = total_loss.item()
        train_epoch_durations.append(time.time() - start_time)

        if not silent:
            train_avg_acc, individual_accuraccy = train_acc.get_accuracy()
            print(f"--- " +
                f"Epoch [{epoch+1}/{num_epochs}], " +
                f"Remaining time: {remain_time}, " +
                f"Learning Rate: {(optimizer.param_groups[0]['lr']):.6f}" +
                f" ---")
            print(f"\tTrain Loss: {total_loss.item():.4f}, " +
                f"Train Accuracy: {(train_avg_acc * 100):.2f}%")
            train_acc.print_accuracy_table()


        if validation_data is not None and validation_interval > 1 and epoch % validation_interval == validation_interval - 1:
            start_time = time.time()
            validation_was_runned = True
            val_loss, val_acc = validation_epoch(model, validation_data, cross_entropy_criterion)
            final_loss = val_loss.item()
            validation_epoch_durations.append(time.time() - start_time)

            if not silent:
                log_validation_info(val_acc, total_loss, val_loss)



        scheduler.step(final_loss)

        train_stats.addEpoch(TrainStatEpoch(
            learning_rate=optimizer.param_groups[0]['lr'],
            train=TrainStatEpochResult(
                loss=total_loss.item(),
                accuracy=train_acc.get_correct_over_total(),
                duration=train_epoch_durations[-1]
            ),
            validation=None if not validation_was_runned else TrainStatEpochResult(
                loss=val_loss.item(),
                accuracy=val_acc.get_correct_over_total(),
                duration=validation_epoch_durations[-1]
            ),
            confusing_pairs=confused_sets.confusing_pair,
            batch_size=train_data.batch_size,
            weights_balance=None if weights_balance is None else weights_balance.tolist()
        ))


        # Estimating remaining time
        remain_epoch: int = num_epochs - (epoch + 1)
        estimated_train_epoch_total_duration: int = 0
        if len(train_epoch_durations) > 0:
            estimated_train_epoch_total_duration = (sum(train_epoch_durations) / len(train_epoch_durations)) * remain_epoch
        estimated_validation_epoch_total_duration: int = 0
        if len(validation_epoch_durations) > 0:
            estimated_validation_epoch_total_duration = (sum(validation_epoch_durations) / len(validation_epoch_durations)) * (remain_epoch / validation_interval)
        estimated_total_duration = estimated_train_epoch_total_duration + estimated_validation_epoch_total_duration
        remain_time = time.strftime('%H:%M:%S', time.gmtime(estimated_total_duration))

    if validation_data is not None and not validation_was_runned:
        start_time = time.time()
        val_loss, val_acc = validation_epoch(model, validation_data, cross_entropy_criterion)
        final_loss = val_loss.item()
        validation_epoch_durations.append(time.time() - start_time)

        if not silent:
            log_validation_info(val_acc, total_loss, val_loss)

    train_stats.final_accuracy = TrainStatEpochResult(
        loss=final_loss,
        accuracy=train_acc.get_correct_over_total(),
        duration=validation_epoch_durations[-1]
    )

    total_time = time.time() - total_time
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    train_stats.total_duration = total_time

    return train_stats
