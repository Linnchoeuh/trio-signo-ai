import random
from dataclasses import dataclass

import torch

from src.datasamples import DataSamples, DataSamplesInfo


@dataclass
class ConfusedSets:
    confusing_pair: dict[int, int]
    confusing_samples: dict[int, list[torch.Tensor]]

    def __init__(self, confusing_pair: dict[str, str], data_samples: DataSamples):
        self.confusing_pair = dict()
        self.confusing_samples = dict()

        for key, value in confusing_pair.items():
            key = data_samples.info.label_map[key]
            value = data_samples.info.label_map[value]
            self.confusing_pair[key] = value
            self.confusing_samples[key] = None
            self.confusing_samples[value] = None

        # print(self.confusing_pair, self.confusing_samples)

        for key in self.confusing_samples.keys():
            self.confusing_samples[key] = data_samples.getTensorsFromLabelId(key)

    def getPositiveSamples(self, anchor_labels: torch.Tensor) -> torch.Tensor:
        positive_samples: list[torch.Tensor] = []
        # print(anchor_labels)
        for anchor_label in anchor_labels:
            positive_samples.append(self.confusing_samples[anchor_label.item()][random.randint(0, len(self.confusing_samples[anchor_label.item()]) - 1)])

        return torch.stack(positive_samples)

    def getNegativeSamples(self, anchor_labels: torch.Tensor) -> torch.Tensor:
        negative_samples: list[torch.Tensor] = []
        for anchor_label in anchor_labels:
            negative_label = self.confusing_pair[anchor_label.item()]
            negative_samples.append(self.confusing_samples[negative_label][random.randint(0, len(self.confusing_samples[negative_label]) - 1)])

        return torch.stack(negative_samples)
