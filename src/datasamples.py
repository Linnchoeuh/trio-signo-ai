import cbor2
import random
import io
from dataclasses import dataclass, fields

from src.gesture import *
from src.datasample import *

@dataclass
class TrainTensors:
    train: tuple[torch.Tensor, torch.Tensor]
    confusion: tuple[torch.Tensor, torch.Tensor]
    validation: tuple[torch.Tensor, torch.Tensor]

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
        # print(len(cls.samples))

        sample_label_id: int = 0
        for labeled_samples in dict_sample:
            current_label: str = cls.info.labels[sample_label_id]
            # print(f"\rLoading label: ({current_label}) {cls.info.label_map[current_label]}/{len(cls.info.labels)}", end="", flush=True)
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

    def toTensors(self, split_ratio: float = 0, confused_label: list[int] = [], include_confused_label_in_train: bool = True) -> TrainTensors:
        samples_out_of_dict: list[list[list[float]]] = []

        def convert_to_tensor(samples: list[list[float]]) -> torch.Tensor | None:
            if len(samples) == 0:
                return None
            sub_step: list[torch.Tensor] = []
            for sample in samples:
                sub_step.append(DataSample2.unflat("", sample, self.valid_fields).to_tensor(self.info.memory_frame, self.valid_fields))
            return torch.stack(sub_step)

        for sample_from_label in self.samples:
            tmp: list[float] = list(sample_from_label.values())
            random.shuffle(tmp)
            samples_out_of_dict.append(tmp)

        train_in: list[list[float]] = []
        train_out: list[int] = []
        confusion_in: list[list[float]] = []
        confusion_out: list[int] = []
        validation_in: list[list[float]] = []
        validation_out: list[int] = []

        for i in range(len(samples_out_of_dict)):
            split_index = int(len(samples_out_of_dict[i]) * (1 - split_ratio))
            if i in confused_label:
                confusion_in += samples_out_of_dict[i][:split_index]
                confusion_out += [i] * split_index
            if include_confused_label_in_train or i not in confused_label:
                train_in += samples_out_of_dict[i][:split_index]
                train_out += [i] * split_index
            validation_in += samples_out_of_dict[i][split_index:]
            validation_out += [i] * (len(samples_out_of_dict[i]) - split_index)

        return TrainTensors(
            train=(convert_to_tensor(train_in), torch.tensor(train_out)),
            confusion=(convert_to_tensor(confusion_in), torch.tensor(confusion_out)),
            validation=(convert_to_tensor(validation_in), torch.tensor(validation_out))
        )

    def getTensorsFromLabel(self, label: str, device: torch.device = torch.device("cpu")) -> list[torch.Tensor]:
        label_id = self.info.label_map.get(label)
        tensors: list[torch.Tensor] = []
        if label_id is None:
            raise ValueError(f"Label {label} is not registered in label_map of this DataSamples class.")

        for sample in self.samples[label_id].values():
            tensors.append(DataSample2.unflat(label, sample, self.valid_fields).to_tensor(self.info.memory_frame, self.valid_fields, device))

        return tensors

    def getTensorsFromLabelId(self, label_int: int, device: torch.device = torch.device("cpu")) -> list[torch.Tensor]:
        return self.getTensorsFromLabel(self.info.labels[label_int], device)

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

class DataSamplesTensors:
    info: DataSamplesInfo
    samples: list[torch.Tensor]
    sample_count: int
    valid_fields: list[str] = []

    def __init__(self, info: DataSamplesInfo):
        self.info = info
        self.sample_count = 0

        self.valid_fields = info.active_gestures.getActiveFields()
        self.samples = []
        while len(self.samples) < len(info.labels):
            self.samples.append(None)


    @classmethod
    def fromDict(cls, json_data):
        cls = cls(info=DataSamplesInfo.fromDict(json_data['info']))
        dict_sample: list[list[list[float]]] = json_data['samples']
        # print(len(cls.samples))

        for sample_label_id in range(len(dict_sample)):
            current_label: str = cls.info.labels[sample_label_id]

            tensor_samples: list[torch.Tensor] = []
            for sample in dict_sample[sample_label_id]:
                tensor_samples.append(DataSample2.unflat(current_label, sample, cls.valid_fields).to_tensor(cls.info.memory_frame, cls.valid_fields))
            del dict_sample[sample_label_id]
            dict_sample.insert(0, None)
            cls.samples[sample_label_id] = torch.stack(tensor_samples)
            del tensor_samples
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
    def fromCborDecoder(cls, fd: io.BufferedReader, decoder: cbor2.CBORDecoder):
        info: DataSamplesInfo = None
        samples: list[torch.Tensor] = []

        valid_fields: list[str] = []

        # Decode the CBOR map header
        initial_byte = decoder.read(1)[0]  # Read the first byte
        major_type = initial_byte >> 5     # Extract the major type (should be 5 for maps)
        if major_type != 5:
            raise ValueError("The first item is not a CBOR map!")

        # Extract the size of the map
        map_size = initial_byte & 0x1F
        if map_size == 31:  # Indefinite-length maps not supported in this example
            raise ValueError("Indefinite-length maps are not supported in this example!")

        for _ in range(map_size):
            key = decoder.decode()  # Decode the key
            # print(f"Processing field: {key}")

            if key == "info":
                data = decoder.decode()
                info = DataSamplesInfo.fromDict(data)
                valid_fields = info.active_gestures.getActiveFields()
            elif key == "samples":
                header = decoder.read(1)[0]  # Read the array header
                major_type = header >> 5
                additional_info = header & 0x1F

                if major_type != 4:
                    raise ValueError(f"Expected array for 'samples', got type {major_type}.")

                # Decode the size of the array based on the additional information
                if additional_info < 24:
                    list_size = additional_info  # Directly encoded size
                elif additional_info == 24:
                    list_size = decoder.read(1)[0]  # Next byte contains the size
                elif additional_info == 25:
                    list_size = int.from_bytes(decoder.read(2), "big")  # 16-bit size
                elif additional_info == 26:
                    list_size = int.from_bytes(decoder.read(4), "big")  # 32-bit size
                elif additional_info == 27:
                    list_size = int.from_bytes(decoder.read(8), "big")  # 64-bit size
                else:
                    raise ValueError("Indefinite-length arrays are not supported.")

                print(f"\r\033[KLoading trainset samples: 0/{list_size}", end="", flush=True)
                for i in range(list_size):
                    sample_from_label = decoder.decode()
                    tensor_samples: list[torch.Tensor] = []
                    for sample in sample_from_label:
                        tensor_samples.append(DataSample2.unflat("", sample, valid_fields).to_tensor(info.memory_frame, valid_fields))
                    del sample_from_label
                    samples.append(torch.stack(tensor_samples))
                    del tensor_samples
                    print(f"\r\033[KLoading trainset samples: {i + 1}/{list_size} [{info.labels[i]}]", end="", flush=True)

        cls = cls(info=info)
        cls.samples = samples
        cls.getNumberOfSamples()
        return cls

    @classmethod
    def fromCborFile(cls, file_path: str):
        with open(file_path, 'rb') as f:
            decoder: cbor2.CBORDecoder = cbor2.CBORDecoder(f)
            return cls.fromCborDecoder(f, decoder)

    def getNumberOfSamples(self):
        self.sample_count = sum([len(label_samples) for label_samples in self.samples])
        return self.sample_count

    # def toDict(self) -> dict:
    #     # self.sample_count = self.getNumberOfSamples()

    #     samples: list[list[list[float]]] = []
    #     for labeled_samples in self.samples:
    #         tmp: list[list[float]] = []
    #         for sample in labeled_samples.values():
    #             # samples[-1].append(sample.flat(self.valid_fields))
    #             tmp.append(sample)
    #         samples.append(tmp)

    #     return {
    #         "info": self.info.toDict(),
    #         "samples": samples
    #     }

    # def toJsonFile(self, file_path: str, indent: int | str | None = 0):
    #     with open(file_path, 'w', encoding="utf-8") as f:
    #         json.dump(self.toDict(), f, indent=indent)

    # def toCbor(self) -> bytes:
    #     return cbor2.dumps(self.toDict())

    # def toCborFile(self, file_path: str):
    #     with open(file_path, 'wb') as f:
    #         f.write(self.toCbor())

    def toTensors(self, split_ratio: float = 0, confused_label: list[int] = [], include_confused_label_in_train: bool = True) -> TrainTensors:

        train_in: list[torch.Tensor] = []
        train_out: list[int] = []
        confusion_in: list[torch.Tensor] = []
        confusion_out: list[int] = []
        validation_in: list[torch.Tensor] = []
        validation_out: list[int] = []

        for i in range(len(self.samples)):
            indices = torch.randperm(self.samples[i].size(0))  # Random permutation of row indices
            shuffled_tensor: torch.Tensor = self.samples[i][indices]

            split_index = int(shuffled_tensor.shape[0] * (1 - split_ratio))

            if i in confused_label:
                confusion_in += shuffled_tensor[:split_index]
                confusion_out += [i] * split_index

            if include_confused_label_in_train or i not in confused_label:
                train_in += shuffled_tensor[:split_index]
                train_out += [i] * split_index

            validation_in += shuffled_tensor[split_index:]
            validation_out += [i] * (shuffled_tensor.shape[0] - split_index)

            del shuffled_tensor

        def to_stack(samples: list[torch.Tensor]) -> torch.Tensor | None:
            if len(samples) == 0:
                return None
            return torch.stack(samples)

        return TrainTensors(
            train=(to_stack(train_in), torch.tensor(train_out)),
            confusion=(to_stack(confusion_in), torch.tensor(confusion_out)),
            validation=(to_stack(validation_in), torch.tensor(validation_out))
        )

    def getTensorsFromLabelId(self, label_int: int, device: torch.device = torch.device("cpu")) -> list[torch.Tensor]:
        return list(torch.unbind(self.samples[label_int], dim=0))

    def getTensorsFromLabel(self, label: str, device: torch.device = torch.device("cpu")) -> list[torch.Tensor]:
        return self.getTensorsFromLabelId(self.info.label_map[label], device)

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
