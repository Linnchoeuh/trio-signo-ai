import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import time
from dataclasses import dataclass

from mediapipe.tasks.python.components.containers.landmark import Landmark

from src.datasample import *

@dataclass
class ModelInfoV1:
    labels: list[str]
    memory_frame: int
    model_version: str = "v1"
    model_name: str = ""


FRAME_SIZE = 15


class SignRecognizerV1(nn.Module):
    def __init__(self, output_size: int, nb_of_memory_frame: int = FRAME_SIZE):
        super(SignRecognizerV1, self).__init__()
        self.input_neurons = nb_of_memory_frame * NEURON_CHUNK
        self.fc1 = nn.Linear(self.input_neurons, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loadModel(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def use(self, input: list[float]):
        # for i in range(len(input)):
        #     input[i] = round(input[i], 3)
        while len(input) < self.input_neurons:
            input.append(0)
        self.eval()
        input_tensor = torch.tensor(input, dtype=torch.float32)
        with torch.no_grad():
            logits = self(input_tensor)
        probabilities = F.softmax(logits, dim=0)
        return torch.argmax(probabilities, dim=0).item()

    def trainModel(self, train_data: TrainData, model_name: str = None, num_epochs: int = 20) -> str:
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

        dataset = CustomDataset(train_data.get_input_data(), train_data.get_output_data(), self.input_neurons)
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

        if model_name is None:
            model_name = f"model_{time.strftime('%d-%m-%Y_%H-%M-%S')}.pth"
        elif not model_name.endswith(".pth"):
            model_name += ".pth"

        torch.save(self.state_dict(), model_name)
        return model_name

    # def add_frame(self, hand_landmark: HandLandmarkerResult):
    #     try:
    #         sample: DataSample = DataSample.from_handlandmarker(hand_landmark)
    #         self.input_data += sample.samples_to_1d_array()
    #     except:
    #         for i in range(NEURON_CHUNK):
    #             self.input_data.append(0)
    #     while len(self.input_data) > self.input_neurons:
    #         self.input_data.pop(0)
