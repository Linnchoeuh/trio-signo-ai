import cbor2
import random
from dataclasses import dataclass, fields

from src.gesture import *
from src.datasample import *

@dataclass
class DataSamplesInfo:
    labels: list[str]
    label_map: dict[str, int]
    memory_frame: int
    active_gestures: ActiveGestures
    one_side: bool


    def __init__(self, labels: list[str], memory_frame: int, active_gestures: ActiveGestures = ALL_GESTURES, label_map: dict[str, int] = None, one_side: bool = False):
        self.labels = labels
        self.memory_frame = memory_frame
        self.active_gestures = active_gestures
        self.label_map = label_map
        self.one_side = one_side


        if self.label_map is None:
            self.label_map = {label: i for i, label in enumerate(labels)}
        else:
            for label in labels:
                if label not in self.label_map:
                    raise ValueError(f"Label {label} not found in label_map")

    @classmethod
    def fromDict(cls, data: dict):
        active_gest_dict: dict = data.get('active_gestures', None)
        active_gest: ActiveGestures = None
        if active_gest_dict is not None:
            active_gest = ActiveGestures(**active_gest_dict)
        return cls(
            labels=data['labels'],
            memory_frame=data['memory_frame'],
            active_gestures=active_gest,
            label_map=data.get('label_map', None),
            one_side=data.get('one_side', False)
        )

    def toDict(self):
        active_gestures = self.active_gestures
        if self.active_gestures is not None:
            active_gestures = self.active_gestures.to_dict()
        return {
            'labels': self.labels,
            'memory_frame': self.memory_frame,
            'active_gestures': active_gestures,
            'label_map': self.label_map,
            'one_side': self.one_side
        }

class DataSamples:
    info: DataSamplesInfo
    samples: list[dict[int, list[float]]]
    sample_count: int

    def __init__(self, info: DataSamplesInfo):
        self.info = info
        self.sample_count = 0

        self.valid_fields: list[str] = info.active_gestures.getActiveFields()
        self.samples = []
        while len(self.samples) < len(info.labels):
            self.samples.append({})


    @classmethod
    def fromDict(cls, json_data):
        cls = cls(info=DataSamplesInfo.fromDict(json_data['info']))
        dict_sample: list[list[list[float]]] = json_data['samples']
        print(len(cls.samples))

        sample_label_id: int = 0
        for labeled_samples in dict_sample:
            current_label: str = cls.info.labels[sample_label_id]
            print(f"\rLoading label: ({current_label}) {cls.info.label_map[current_label]}/{len(cls.info.labels)}", end="", flush=True)
            for sample in labeled_samples:
                # new_datasample: DataSample2 = DataSample2.unflat(current_label, sample, cls.valid_fields)
                cls.samples[sample_label_id][id(sample)] = sample
                # print(len(cls.samples[sample_label_id]))
            sample_label_id += 1
        cls.getNumberOfSamples()
        return cls

    @classmethod
    def fromJsonFile(cls, file_path: str):
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return cls.fromDict(data)

    @classmethod
    def fromCbor(cls, cbor_data):
        return cls.fromDict(cbor2.loads(cbor_data))

    @classmethod
    def fromCborFile(cls, file_path: str):
        with open(file_path, 'rb') as f:
            data = f.read()
        return cls.fromCbor(data)

    def getNumberOfSamples(self):
        self.sample_count = sum([len(label_samples) for label_samples in self.samples])
        return self.sample_count

    def toDict(self) -> dict:
        # self.sample_count = self.getNumberOfSamples()

        samples: list[list[list[float]]] = []
        for labeled_samples in self.samples:
            tmp: list[list[float]] = []
            for sample in labeled_samples.values():
                # samples[-1].append(sample.flat(self.valid_fields))
                tmp.append(sample)
            samples.append(tmp)

        return {
            "info": self.info.toDict(),
            "samples": samples
        }

    def toJsonFile(self, file_path: str, indent: int | str | None = 0):
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(self.toDict(), f, indent=indent)

    def toCbor(self) -> bytes:
        return cbor2.dumps(self.toDict())

    def toCborFile(self, file_path: str):
        with open(file_path, 'wb') as f:
            f.write(self.toCbor())

    # def toTensors(self, device: torch.device = torch.device("cpu"), split_ratio: float = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     all_samples: dict[int, DataSample2] = {}

    #     for label_samples in self.samples:
    #         all_samples.update(label_samples)

    #     keys = random.sample(list(all_samples.keys()), len(all_samples))

    #     input_data: list[torch.Tensor] = []
    #     output_data: list[int] = []

    #     while len(keys) > 0:
    #         key = keys.pop()
    #         output_data.append(self.info.label_map[all_samples[key].label])
    #         input_data.append(all_samples[key].to_tensor(self.info.memory_frame, self.valid_fields, device))

    #     if split_ratio > 0:
    #         split_index = int(len(input_data) * split_ratio)
    #         return torch.stack(input_data[:split_index]), torch.tensor(output_data[:split_index]), torch.stack(input_data[split_index:]), torch.tensor(output_data[split_index:])
    #     return torch.stack(input_data), torch.tensor(output_data), None, None

    def toTensors(self, device: torch.device = torch.device("cpu"), split_ratio: float = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_samples: dict[int, tuple[list[float], int]] = {}

        for i in range(len(self.samples)):
            for _id, sample in self.samples[i].items():
                all_samples[_id] = (sample, i)

        keys = random.sample(list(all_samples.keys()), len(all_samples))

        input_data: list[torch.Tensor] = []
        output_data: list[int] = []

        while len(keys) > 0:
            key = keys.pop()
            output_data.append(all_samples[key][1])
            input_data.append(DataSample2.unflat("", all_samples[key][0], self.valid_fields).to_tensor(self.info.memory_frame, self.valid_fields, device))

        if split_ratio > 0:
            split_index = int(len(input_data) * split_ratio)
            return torch.stack(input_data[:split_index]), torch.tensor(output_data[:split_index]), torch.stack(input_data[split_index:]), torch.tensor(output_data[split_index:])
        return torch.stack(input_data), torch.tensor(output_data), None, None

    def addDataSample(self, data_sample: DataSample2):
        # Get or cache label_id
        label_id = self.info.label_map.get(data_sample.label)
        if label_id is None:
            raise ValueError(f"Label {data_sample.label} is not registered in label_map of this DataSamples class.")

        if self.info.one_side:
            data_sample.move_to_one_side()

        flat_list: list[float] = data_sample.flat(self.valid_fields)
        # Use self.sample_count as a unique identifier instead of len(self.samples[label_id])
        self.samples[label_id][id(flat_list)] = flat_list
        # self.samples[label_id].add(sample_data)

        # Increment the overall sample count
        self.sample_count += 1

    def addDataSamples(self, data_samples: list[DataSample2]):
        for data_sample in data_samples:
            # print(type(data_sample))
            self.addDataSample(data_sample)

    # def getInputData(self) -> list[list[float]]:
    #     """Transform the trainset data into a 1 dimension array
    #     where each list[float] is a sample

    #     Returns:
    #         list[list[float]]: _description_
    #     """
    #     samples: list[list[float]] = []
    #     for label_sorted_samples in self.samples:
    #         for sample in label_sorted_samples: # Get all the sample stored in the "set"
    #             # Convert the "tuple[int, tuple[float]]" to "list[float]"
    #             # We discard the id of the sample and convert the "tuple[float]" to "list[float]"
    #             samples.append(list(sample[1]))
    #             # samples.append(list(sample))
    #     return samples

    # def get_output_data(self) -> list[int]:
    #     labels: list[int] = []
    #     for i in range(len(self.samples)):
    #         labels += [i] * len(self.samples[i])
    #     return labels

    # def splitTrainset(self, ratio: float = 0.8) -> tuple['TrainData2', 'TrainData2']:
    #     """Split the trainset into two trainset

    #     Args:
    #         ratio (float, optional): Ratio of the first trainset. Defaults to 0.8.

    #     Returns:
    #         tuple[TrainData2, TrainData2]: The two trainset
    #     """
    #     trainset1 = TrainData2(info=copy.deepcopy(self.info))
    #     trainset2 = TrainData2(info=copy.deepcopy(self.info))

    #     for i in range(len(self.samples)):

    #         total_label_samples = len(self.samples[i])
    #         label_sample: set[tuple[int, tuple[float]]] = list(self.samples[i])

    #         while len(label_sample) / total_label_samples > ratio:
    #             trainset2.samples[i].add(
    #                 label_sample.pop(random.randint(0, len(label_sample) - 1))
    #             )
    #         trainset1.samples[i] = set(label_sample)

    #     trainset1.getNumberOfSamples()
    #     trainset2.getNumberOfSamples()
    #     return trainset1, trainset2

    def getClassWeights(self, balance_weight: bool = True, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        weigths: list[float] = []
        if balance_weight:
            smallest_class = len(self.samples[0])
            for sample in self.samples:
                if len(sample) < smallest_class:
                    smallest_class = len(sample)
            for sample in self.samples:
                weigths.append(smallest_class / len(sample))
        else:
            for sample in self.samples:
                weigths.append(1)
        total = sum(weigths)
        weigths = [weight / total for weight in weigths]
        # print(weigths)
        return torch.tensor(weigths, dtype=torch.float32, device=device)
