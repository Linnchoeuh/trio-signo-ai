import json
import time
import os
import glob
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.datasample import *

@dataclass
class ModelInfo(TrainDataInfo):
    layers: list[int]
    name: str
    mode_arch: str = "v1"

    @classmethod
    def build(cls, nb_of_memory_frame: int, active_gestures: ActiveGestures, output_labels: list[str], intermediate_layers: list[int] = [128, 64], name: str = None) -> 'ModelInfo':
        if name is None:
            name = f"model_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
        layers: list[int] = [nb_of_memory_frame * len(active_gestures.getActiveFields()) * 3] + intermediate_layers + [len(output_labels)]
        label_map: dict[str, int] = {label: i for i, label in enumerate(output_labels)}
        return cls(
            labels=output_labels,
            label_map=label_map,
            memory_frame=nb_of_memory_frame,
            active_gestures=active_gestures,
            layers=layers,
            name=name
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelInfo':
        data["active_gestures"] = ActiveGestures(**data["active_gestures"])
        return cls(**data)

    @classmethod
    def from_json_file(cls, file_path: str) -> 'ModelInfo':
        with open(file_path, 'r', encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self):
        _dict: dict = super(ModelInfo, self).to_dict()
        _dict["layers"] = self.layers
        _dict["name"] = self.name
        _dict["mode_arch"] = self.mode_arch
        return _dict

    def to_json_file(self, file_path: str, indent: int = 4):
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)

@dataclass
class ModelEpochResult:
    model_instance: 'SignRecognizerV1'
    train_loss: float
    validation_loss: float
    loss_diff: float
    mean_loss: float
    epoch: int

class AccuracyCalculator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.correct_per_class = None
        self.total_per_class = None
        self.reset()

    def reset(self):
        self.correct_per_class = [0] * self.num_classes # To track correct predictions per class
        self.total_per_class = [0] * self.num_classes # To track total samples per class

    def calculate_accuracy(self, outputs: torch.Tensor, labels):
        # Get predictions
        _, predictions = torch.max(outputs, 1)  # Predicted class indices

        # Update correct and total counts for each class
        for label in range(self.num_classes):
            self.correct_per_class[label] += ((predictions == label) & (labels == label)).sum().item()
            self.total_per_class[label] += (labels == label).sum().item()

    def get_accuracy(self) -> tuple[float, dict]:
        avg_accuracy = sum(self.correct_per_class) / sum(self.total_per_class) if sum(self.total_per_class) > 0 else 0
        return (avg_accuracy, [correct / total if total > 0 else 0 for correct, total in zip(self.correct_per_class, self.total_per_class)])

class SignRecognizerV1(nn.Module):
    def __init__(self, model_info: ModelInfo, device: torch.device = torch.device("cpu")):
        super(SignRecognizerV1, self).__init__()

        self.info = model_info
        self.fcs = nn.ModuleList()
        self.device: torch.device = device

        for i in range(len(model_info.layers) - 1):
            self.fcs.append(nn.Linear(model_info.layers[i], model_info.layers[i+1]))
            if i < len(model_info.layers) - 2:
                self.fcs.append(nn.Dropout(0.3))

        self.to(self.device)

    @classmethod
    def loadModelFromDir(cls, model_dir: str, device: torch.device = torch.device("cpu")):
        json_files = glob.glob(f"{model_dir}/*.json")
        if len(json_files) == 0:
            raise FileNotFoundError(f"No .json file found in {model_dir}")
        cls = SignRecognizerV1(ModelInfo.from_json_file(json_files[0]), device=device)

        pth_files = glob.glob(f"{model_dir}/*.pth")
        if len(pth_files) == 0:
            raise FileNotFoundError(f"No .pth file found in {model_dir}")
        cls.loadPthFile(pth_files[0])

        return cls

    def loadPthFile(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=self.device))

    def saveModel(self, model_name: str = None):
        if model_name is None:
            model_name = self.info.name
        if model_name.endswith(".pth"):
            model_name = model_name[:-4]
        if model_name.endswith(".json"):
            model_name = model_name[:-5]

        os.makedirs(model_name, exist_ok=True)
        full_name = f"{model_name}/{model_name}"
        torch.save(self.state_dict(), full_name + ".pth")
        self.info.to_json_file(full_name + ".json")

    def forward(self, x):
        for i in range(len(self.fcs) - 1):
            x = torch.relu(self.fcs[i](x))
        return self.fcs[-1](x)

    def use(self, input: list[float]):
        while len(input) < self.info.layers[0]:
            input.append(0)
        self.eval()
        input_tensor = torch.tensor(input, dtype=torch.float32)
        with torch.no_grad():
            logits = self(input_tensor)
        probabilities = F.softmax(logits, dim=0)
        return torch.argmax(probabilities, dim=0).item()

    def trainModel(self, train_data: TrainData2, num_epochs: int = 20, validation_data: TrainData2 = None) -> str:
        model_epochs: list[ModelEpochResult] = []
        class CustomDataset(Dataset):
            def __init__(self, input: list[list[float]], output: list[int], model_input_neuron: int):
                if len(input) != len(output):
                    raise ValueError("Input and output data must have the same length")

                self.input: list[list[float]] = input
                self.output: list[int] = output
                self.model_input_neuron = model_input_neuron
                # print(len(input), len(output))

            def __len__(self):
                return len(self.input)

            def __getitem__(self, idx):
                input_data: list[float] = self.input[idx]
                # print(input_data)
                while len(input_data) < self.model_input_neuron:
                    input_data.append(0)
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
                label_tensor = torch.tensor(self.output[idx], dtype=torch.long)
                return input_tensor, label_tensor

        dataset: CustomDataset = CustomDataset(train_data.get_input_data(), train_data.get_output_data(), self.info.layers[0])
        dataloader: DataLoader = DataLoader(dataset, batch_size=64, shuffle=True)
        if validation_data is not None:
            validation_set: CustomDataset = CustomDataset(validation_data.get_input_data(), validation_data.get_output_data(), self.info.layers[0])
            validation_dataloader: DataLoader = DataLoader(validation_set, batch_size=64, shuffle=True)

        criterion = nn.CrossEntropyLoss(train_data.get_class_weights()).to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        train_accuracy_calculator = AccuracyCalculator(len(self.info.labels))
        validation_accuracy_calculator = AccuracyCalculator(len(self.info.labels))


        start_time = time.time()
        for epoch in range(num_epochs):
            train_accuracy_calculator.reset()
            val_los: float = 0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_accuracy_calculator.calculate_accuracy(outputs, labels)

            val_loss = loss.item()

            lr = optimizer.param_groups[0]['lr']

            print(f"--- " +
                f"Epoch [{epoch+1}/{num_epochs}], " +
                f"Remaining time: {time.strftime('%H:%M:%S', time.gmtime(((time.time() - start_time) / (epoch+1)) * (num_epochs - epoch - 1)))}, " +
                f"Learning Rate: {lr:.6f}" +
                f" ---")
            train_avg_acc, train_accuracy = train_accuracy_calculator.get_accuracy()
            print(f"\tTrain Loss: {loss.item():.4f}, " +
                f"Train Accuracy: {(train_avg_acc * 100):.2f}%")
            for i, label in enumerate(self.info.labels):
                print(f"\t\t{label}: {(train_accuracy[i] * 100):.2f}%")
            if validation_data is not None:
                total_correct = 0
                validation_loss = None
                validation_accuracy_calculator.reset()
                for inputs, labels in validation_dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self(inputs)
                    validation_loss = criterion(outputs, labels)

                    validation_accuracy_calculator.calculate_accuracy(outputs, labels)

                validation_avg_acc, validation_accuracy = validation_accuracy_calculator.get_accuracy()
                print(f"\tValidation Loss: {validation_loss.item():.4f}, " +
                    f"Validation accuracy: {(validation_avg_acc * 100):.2f}%")
                for i, label in enumerate(self.info.labels):
                    print(f"\t\t{label}: {(validation_accuracy[i] * 100):.2f}%")

                loss_diff: float = abs(loss.item() - validation_loss.item())
                mean_loss: float = (loss.item() + validation_loss.item()) / 2

                val_loss += validation_loss.item()

                print(f"\tLoss Diff: {loss_diff:.4f}, Mean Loss: {mean_loss:.4f}")

            scheduler.step(val_loss)

            model_epochs.append(ModelEpochResult(copy.deepcopy(self), loss.item(),
                                                 validation_loss.item() if validation_loss is not None else None,
                                                 loss_diff, mean_loss, epoch+1))

        """
        model_epochs.sort(key=lambda x: x.loss_diff)
        # print(model_epochs)
        while len(model_epochs) > 5:
            model_epochs.pop(-1)
        # for model_epoch in model_epochs:
        #     print(f"Epoch {model_epoch.epoch}: Loss diff: {model_epoch.loss_diff:.4f}, Mean Loss: {model_epoch.mean_loss:.4f}")
        model_epochs.sort(key=lambda x: x.validation_loss)
        # print("-----")
        # for model_epoch in model_epochs:
        #     print(f"Epoch {model_epoch.epoch}: Loss diff: {model_epoch.loss_diff:.4f}, Mean Loss: {model_epoch.mean_loss:.4f}")
        # print(model_epochs)
        print("Model Epoch Picked:", model_epochs[0].epoch)
        self = model_epochs[0].model_instance
        """
