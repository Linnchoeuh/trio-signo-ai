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



class SignRecognizerV1(nn.Module):
    def __init__(self, model_info: ModelInfo):
        super(SignRecognizerV1, self).__init__()

        self.info = model_info
        self.fcs = nn.ModuleList()

        for i in range(len(model_info.layers) - 1):
            self.fcs.append(nn.Linear(model_info.layers[i], model_info.layers[i+1]))

    @classmethod
    def loadModelFromDir(cls, model_dir: str):
        json_files = glob.glob(f"{model_dir}/*.json")
        if len(json_files) == 0:
            raise FileNotFoundError(f"No .json file found in {model_dir}")
        cls = SignRecognizerV1(ModelInfo.from_json_file(json_files[0]))

        pth_files = glob.glob(f"{model_dir}/*.pth")
        if len(pth_files) == 0:
            raise FileNotFoundError(f"No .pth file found in {model_dir}")
        cls.loadPthFile(pth_files[0])

        return cls

    def loadPthFile(self, model_path):
        self.load_state_dict(torch.load(model_path))

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
        # for i in range(len(input)):
        #     input[i] = round(input[i], 3)
        while len(input) < self.info.layers[0]:
            input.append(0)
        self.eval()
        input_tensor = torch.tensor(input, dtype=torch.float32)
        with torch.no_grad():
            logits = self(input_tensor)
        probabilities = F.softmax(logits, dim=0)
        return torch.argmax(probabilities, dim=0).item()

    def trainModel(self, train_data: TrainData, num_epochs: int = 20) -> str:
        class CustomDataset(Dataset):
            def __init__(self, input: list[list[float]], output: list[int], model_input_neuron: int):
                if len(input) != len(output):
                    raise ValueError("Input and output data must have the same length")

                self.input: list[list[float]] = input
                self.output: list[int] = output
                self.model_input_neuron = model_input_neuron

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

        dataset = CustomDataset(train_data.get_input_data(), train_data.get_output_data(), self.info.layers[0])
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # print(outputs)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
