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

LABEL_MAP = LabelMap(
    label={
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "J": 9,
        "K": 10,
        "L": 11,
        "M": 12,
        "N": 13,
        "O": 14,
        "P": 15,
        "Q": 16,
        "R": 17,
        "S": 18,
        "T": 19,
        "U": 20,
        "V": 21,
        "W": 22,
        "X": 23,
        "Y": 24,
        "Z": 25,
        "0": 26,
    },
    id={
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
        10: "K",
        11: "L",
        12: "M",
        13: "N",
        14: "O",
        15: "P",
        16: "Q",
        17: "R",
        18: "S",
        19: "T",
        20: "U",
        21: "V",
        22: "W",
        23: "X",
        24: "Y",
        25: "Z",
        26: "0",
    }
)

def landmarks_to_list(landmarks: list[Landmark]):
    return [[landmark.x, landmark.y, landmark.z] for landmark in landmarks]

def LandmarksTo1DArray(landmarks: list[Landmark]):
    return [item for sublist in landmarks_to_list(landmarks) for item in sublist]

class LSFAlphabetRecognizer(nn.Module):
    def __init__(self):
        super(LSFAlphabetRecognizer, self).__init__()
        self.fc1 = nn.Linear(63, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 27)

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
