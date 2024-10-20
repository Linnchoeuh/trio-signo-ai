from dataclasses import dataclass, fields
from mediapipe.tasks.python.vision.hand_landmarker import *
from mediapipe.tasks.python.components.containers.landmark import *
from src.rot_3d import *
import random
import math
import json
import cbor2

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

POINTS = 3
DATA_POINTS = 21
NEURON_CHUNK = (DATA_POINTS * POINTS)

@dataclass
class GestureData:
    wrist: list[float, float, float] # [x, y, z]
    thumb_cmc: list[float, float, float] # [x, y, z]
    thumb_mcp: list[float, float, float]
    thumb_ip: list[float, float, float]
    thumb_tip: list[float, float, float]
    index_mcp: list[float, float, float]
    index_pip: list[float, float, float]
    index_dip: list[float, float, float]
    index_tip: list[float, float, float]
    middle_mcp: list[float, float, float]
    middle_pip: list[float, float, float]
    middle_dip: list[float, float, float]
    middle_tip: list[float, float, float]
    ring_mcp: list[float, float, float]
    ring_pip: list[float, float, float]
    ring_dip: list[float, float, float]
    ring_tip: list[float, float, float]
    pinky_mcp: list[float, float, float]
    pinky_pip: list[float, float, float]
    pinky_dip: list[float, float, float]
    pinky_tip: list[float, float, float]

    @classmethod
    def from_landmark_list(cls, landmarks: list[Landmark]):
        return cls(
            wrist=[landmarks[0].x, landmarks[0].y, landmarks[0].z],
            thumb_cmc=[landmarks[1].x, landmarks[1].y, landmarks[1].z],
            thumb_mcp=[landmarks[2].x, landmarks[2].y, landmarks[2].z],
            thumb_ip=[landmarks[3].x, landmarks[3].y, landmarks[3].z],
            thumb_tip=[landmarks[4].x, landmarks[4].y, landmarks[4].z],
            index_mcp=[landmarks[5].x, landmarks[5].y, landmarks[5].z],
            index_pip=[landmarks[6].x, landmarks[6].y, landmarks[6].z],
            index_dip=[landmarks[7].x, landmarks[7].y, landmarks[7].z],
            index_tip=[landmarks[8].x, landmarks[8].y, landmarks[8].z],
            middle_mcp=[landmarks[9].x, landmarks[9].y, landmarks[9].z],
            middle_pip=[landmarks[10].x, landmarks[10].y, landmarks[10].z],
            middle_dip=[landmarks[11].x, landmarks[11].y, landmarks[11].z],
            middle_tip=[landmarks[12].x, landmarks[12].y, landmarks[12].z],
            ring_mcp=[landmarks[13].x, landmarks[13].y, landmarks[13].z],
            ring_pip=[landmarks[14].x, landmarks[14].y, landmarks[14].z],
            ring_dip=[landmarks[15].x, landmarks[15].y, landmarks[15].z],
            ring_tip=[landmarks[16].x, landmarks[16].y, landmarks[16].z],
            pinky_mcp=[landmarks[17].x, landmarks[17].y, landmarks[17].z],
            pinky_pip=[landmarks[18].x, landmarks[18].y, landmarks[18].z],
            pinky_dip=[landmarks[19].x, landmarks[19].y, landmarks[19].z],
            pinky_tip=[landmarks[20].x, landmarks[20].y, landmarks[20].z]
        )

    @classmethod
    def from_list(cls, raw_data: list[float]):
        return cls(
            wrist=raw_data[0:3],
            thumb_cmc=raw_data[3:6],
            thumb_mcp=raw_data[6:9],
            thumb_ip=raw_data[9:12],
            thumb_tip=raw_data[12:15],
            index_mcp=raw_data[15:18],
            index_pip=raw_data[18:21],
            index_dip=raw_data[21:24],
            index_tip=raw_data[24:27],
            middle_mcp=raw_data[27:30],
            middle_pip=raw_data[30:33],
            middle_dip=raw_data[33:36],
            middle_tip=raw_data[36:39],
            ring_mcp=raw_data[39:42],
            ring_pip=raw_data[42:45],
            ring_dip=raw_data[45:48],
            ring_tip=raw_data[48:51],
            pinky_mcp=raw_data[51:54],
            pinky_pip=raw_data[54:57],
            pinky_dip=raw_data[57:60],
            pinky_tip=raw_data[60:63]
        )

    def to_list(self):
        raw_coords = []
        for attr in self.__dict__.values():
            # print(attr)
            raw_coords += attr
        return raw_coords

@dataclass
class DataSample:
    label: str
    gestures: list[GestureData]
    label_id: int | None = None
    framerate: int = 30

    @classmethod
    def from_json(cls, json_data: dict, label_id: int = None):
        if label_id is None:
            label_id = json_data['label_id']
        framerate = 30
        if json_data.get("framerate") is not None:
            framerate = json_data['framerate']
        return cls(
            label=json_data['label'],
            label_id=label_id,
            framerate=framerate,
            gestures=[GestureData(**gesture) for gesture in json_data['gestures']]
        )

    @classmethod
    def from_json_file(cls, file_path: str, label_id: int = None):
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json(data, label_id)

    @classmethod
    def from_handlandmarker(cls, hand_landmarks: HandLandmarkerResult, label, label_id):
        return cls(
            label=label,
            label_id=label_id,
            gestures=[GestureData.from_landmark_list(hand_landmarks.hand_world_landmarks[0])]
        )

    def to_json(self):
        return {
            'label': self.label,
            'label_id': self.label_id,
            'gestures': [gesture.__dict__ for gesture in self.gestures]
        }

    def to_json_file(self, file_path: str, indent: bool = False):
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=indent)

    def pushfront_gesture_from_landmarks(self, hand_landmarks: HandLandmarkerResult, allow_empty_frame: bool = True):
        if len(hand_landmarks.hand_world_landmarks) == 0:
            if not allow_empty_frame:
                raise ValueError("Empty frame not allowed")
            else:
                self.gestures.insert(0, GestureData.from_list([0 for _ in range(NEURON_CHUNK)]))
        else:
            self.gestures.insert(0, GestureData.from_landmark_list(hand_landmarks.hand_world_landmarks[0]))

    def samples_to_1d_array(self) -> list[float]:
        raw_data = []
        for gesture in self.gestures:
            # raw_data.append(gesture.to_list())
            raw_data += gesture.to_list()
        return raw_data

    def mirror_sample(self, mirror_x: bool = True, mirror_y: bool = False, mirror_z: bool = False) -> 'DataSample':
        for i in range(len(self.gestures)):
            for field in fields(self.gestures[i]):
                field_value: list[float] = getattr(self.gestures[i], field.name)
                if mirror_x:
                    field_value[0] = -field_value[0]
                if mirror_y:
                    field_value[1] = -field_value[1]
                if mirror_z:
                    field_value[2] = -field_value[2]
                setattr(self.gestures[i], field.name, field_value)
        return self

    def rotate_sample(self, angle_x: float = 0, angle_y: float = 0, angle_z: float = 0) -> 'DataSample':
        for i in range(len(self.gestures)):
            for field in fields(self.gestures[i]):
                field_value: list[float] = getattr(self.gestures[i], field.name)
                field_value = rot_3d_x(field_value, angle_x)
                field_value = rot_3d_y(field_value, angle_y)
                field_value = rot_3d_z(field_value, angle_z)
                setattr(self.gestures[i], field.name, field_value)
        return self

    def randomize_points(self, factor: float = 0.005) -> 'DataSample':
        for i in range(len(self.gestures)):
            for field in fields(self.gestures[i]):
                field_value: list[float] = getattr(self.gestures[i], field.name)
                field_value[0] += clamp((random.random() - 0.5) * factor, -1, 1)
                field_value[1] += clamp((random.random() - 0.5) * factor, -1, 1)
                field_value[2] += clamp((random.random() - 0.5) * factor, -1, 1)
                setattr(self.gestures[i], field.name, field_value)
        return self

    def translate_hand(self, x: float = 0, y: float = 0, z: float = 0) -> 'DataSample':
        for i in range(len(self.gestures)):
            for field in fields(self.gestures[i]):
                field_value: list[float] = getattr(self.gestures[i], field.name)
                field_value[0] += x
                field_value[1] += y
                field_value[2] += z
                setattr(self.gestures[i], field.name, field_value)
        return self

    def deform_hand(self, x: float = 1, y: float = 1, z: float = 1) -> 'DataSample':
        for i in range(len(self.gestures)):
            for field in fields(self.gestures[i]):
                field_value: list[float] = getattr(self.gestures[i], field.name)
                field_value[0] *= x
                field_value[1] *= y
                field_value[2] *= z
                setattr(self.gestures[i], field.name, field_value)
        return self

    def reframe(self, target_frame: int) -> 'DataSample':
        """Change the number of frame to execute the full gesture sequence
        Be careful, if frame are reducedn reincresing the frame will not restore the original gesture

        Args:
            frame (int): Target frame number
        """
        def lerp(a, b, t):
            return [a[i] + (b[i] - a[i]) * t for i in range(len(a))]

        new_gestures: list[GestureData] = []

        if target_frame <= 1:
            raise ValueError("Target frame must be greater than 1")

        for i in range(target_frame):
            progression = i / (target_frame - 1)
            frame_scaled_value = min(progression * (len(self.gestures) - 1), len(self.gestures) - 1)
            # print(i, frame_scaled_value, len(self.gestures))
            start_frame = math.floor(frame_scaled_value)
            end_frame = math.ceil(frame_scaled_value)
            interpolation_coef = frame_scaled_value - start_frame


            new_gestures.append(GestureData(
                wrist=lerp(self.gestures[start_frame].wrist, self.gestures[end_frame].wrist, interpolation_coef),
                thumb_cmc=lerp(self.gestures[start_frame].thumb_cmc, self.gestures[end_frame].thumb_cmc, interpolation_coef),
                thumb_mcp=lerp(self.gestures[start_frame].thumb_mcp, self.gestures[end_frame].thumb_mcp, interpolation_coef),
                thumb_ip=lerp(self.gestures[start_frame].thumb_ip, self.gestures[end_frame].thumb_ip, interpolation_coef),
                thumb_tip=lerp(self.gestures[start_frame].thumb_tip, self.gestures[end_frame].thumb_tip, interpolation_coef),
                index_mcp=lerp(self.gestures[start_frame].index_mcp, self.gestures[end_frame].index_mcp, interpolation_coef),
                index_pip=lerp(self.gestures[start_frame].index_pip, self.gestures[end_frame].index_pip, interpolation_coef),
                index_dip=lerp(self.gestures[start_frame].index_dip, self.gestures[end_frame].index_dip, interpolation_coef),
                index_tip=lerp(self.gestures[start_frame].index_tip, self.gestures[end_frame].index_tip, interpolation_coef),
                middle_mcp=lerp(self.gestures[start_frame].middle_mcp, self.gestures[end_frame].middle_mcp, interpolation_coef),
                middle_pip=lerp(self.gestures[start_frame].middle_pip, self.gestures[end_frame].middle_pip, interpolation_coef),
                middle_dip=lerp(self.gestures[start_frame].middle_dip, self.gestures[end_frame].middle_dip, interpolation_coef),
                middle_tip=lerp(self.gestures[start_frame].middle_tip, self.gestures[end_frame].middle_tip, interpolation_coef),
                ring_mcp=lerp(self.gestures[start_frame].ring_mcp, self.gestures[end_frame].ring_mcp, interpolation_coef),
                ring_pip=lerp(self.gestures[start_frame].ring_pip, self.gestures[end_frame].ring_pip, interpolation_coef),
                ring_dip=lerp(self.gestures[start_frame].ring_dip, self.gestures[end_frame].ring_dip, interpolation_coef),
                ring_tip=lerp(self.gestures[start_frame].ring_tip, self.gestures[end_frame].ring_tip, interpolation_coef),
                pinky_mcp=lerp(self.gestures[start_frame].pinky_mcp, self.gestures[end_frame].pinky_mcp, interpolation_coef),
                pinky_pip=lerp(self.gestures[start_frame].pinky_pip, self.gestures[end_frame].pinky_pip, interpolation_coef),
                pinky_dip=lerp(self.gestures[start_frame].pinky_dip, self.gestures[end_frame].pinky_dip, interpolation_coef),
                pinky_tip=lerp(self.gestures[start_frame].pinky_tip, self.gestures[end_frame].pinky_tip, interpolation_coef),
            ))

        self.gestures = new_gestures
        # print(self.gestures)
        return self

    def round_gesture_coordinates(self, decimal: int = 3) -> 'DataSample':
        for i in range(len(self.gestures)):
            for field in fields(self.gestures[i]):
                field_value: list[float] = getattr(self.gestures[i], field.name)
                field_value = [round(coord, decimal) for coord in field_value]
                setattr(self.gestures[i], field.name, field_value)
        return self


@dataclass
class TrainDataInfo:
    labels: list[str]
    label_map: dict[str, int]
    memory_frame: int

    def __init__(self, labels: list[str], memory_frame: int, label_map: dict[str, int] = None):
        self.labels = labels
        self.memory_frame = memory_frame

        if label_map is None:
            self.label_map = {label: i for i, label in enumerate(labels)}
        else:
            for label in labels:
                if label not in label_map:
                    raise ValueError(f"Label {label} not found in label_map")

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            labels=data['labels'],
            label_map=data['label_map'],
            memory_frame=data['memory_frame']
        )

@dataclass
class TrainData:
    info: TrainDataInfo
    samples: list[set[list[float]]]
    sample_count: int

    def __init__(self, info: TrainDataInfo, samples: list[set[list[float]]] = None):
        self.info = info

        self.test = set()
        if samples is not None:
            self.samples = samples
        else:
            self.samples = []
            while len(self.samples) < len(info.labels):
                self.samples.append(set())
        self.sample_count = sum([len(label_samples) for label_samples in self.samples])

    @classmethod
    def from_json(cls, json_data):
        samples = json_data['samples']
        for i in range(len(samples)):
            for k in range(len(samples[i])):
                samples[i][k] = tuple(samples[i][k])
            samples[i] = set(samples[i])
        return cls(
            info=TrainDataInfo.from_dict(json_data['info']),
            samples=samples
        )

    @classmethod
    def from_json_file(cls, file_path: str):
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json(data)

    @classmethod
    def from_cbor(cls, cbor_data):
        return cls.from_json(cbor2.loads(cbor_data))

    @classmethod
    def from_cbor_file(cls, file_path: str):
        with open(file_path, 'rb') as f:
            data = f.read()
        return cls.from_cbor(data)

    def to_json(self):
        for i in range(len(self.samples)):
            self.samples[i] = list(self.samples[i])
        return {
            'info': self.info.__dict__,
            'samples': self.samples
        }

    def to_json_file(self, file_path: str, indent: bool = False):
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=indent)

    def to_cbor(self):
        return cbor2.dumps(self.to_json())

    def to_cbor_file(self, file_path: str):
        with open(file_path, 'wb') as f:
            f.write(self.to_cbor())

    def add_data_sample(self, data_sample: DataSample, label: str = None):
        if label is None:
            label = data_sample.label

        self.samples[self.info.label_map[label]].add(tuple(data_sample.samples_to_1d_array()))
        self.sample_count += 1
        pass

    def add_data_samples(self, data_samples: list[DataSample]):
        for data_sample in data_samples:
            # print(type(data_sample))
            self.add_data_sample(data_sample, data_sample.label)

    def get_input_data(self) -> list[list[float]]:
        samples: list[list[float]] = []
        for label_sorted_samples in self.samples:
            label_sorted_samples = list(label_sorted_samples)
            for i in range(len(label_sorted_samples)):
                label_sorted_samples[i] = list(label_sorted_samples[i])
            samples += label_sorted_samples
        return samples

    def get_output_data(self) -> list[int]:
        labels: list[int] = []
        for i in range(len(self.samples)):
            labels += [i] * len(self.samples[i])
        return labels
