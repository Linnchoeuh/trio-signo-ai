import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from dataclasses import dataclass
from src.datasample import *

from mediapipe.tasks.python.components.containers.landmark import Landmark

@dataclass
class ModelInfoV1:
    labels: list[str]
    model_version: str = "v2"
    model_name: str = ""

def landmarks_to_list(landmarks: list[Landmark]):
    return [[landmark.x, landmark.y, landmark.z] for landmark in landmarks]

def LandmarksTo1DArray(landmarks: list[Landmark]):
    return [item for sublist in landmarks_to_list(landmarks) for item in sublist]

class SignRecognizerV1(nn.Module):
    def __init__(self, output_size: int):
        super(SignRecognizerV1, self).__init__()
        self.fc1 = nn.Linear(63, 128)
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
