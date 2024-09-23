import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from dataclasses import dataclass

from mediapipe.tasks.python.components.containers.landmark import Landmark

@dataclass
class LabelMap:
    label: dict
    id: dict

@dataclass
class ModelInfoV1:
    model_version: str
    labels: list[str]
    model_name: str = ""

def landmarks_to_list(landmarks: list[Landmark]):
    return [[landmark.x, landmark.y, landmark.z] for landmark in landmarks]

def LandmarksTo1DArray(landmarks: list[Landmark]):
    return [item for sublist in landmarks_to_list(landmarks) for item in sublist]

class LSFAlphabetRecognizerV1(nn.Module):
    def __init__(self, output_size: int):
        super(LSFAlphabetRecognizerV1, self).__init__()
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
