import copy
import math
import json
import random
from dataclasses import dataclass, fields
from mediapipe.tasks.python.vision.hand_landmarker import *
from mediapipe.tasks.python.components.containers.landmark import *
from src.rot_3d import *
import cbor2
import torch

from src.gesture import *

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

# POINTS = 3
# DATA_POINTS = 21
# NEURON_CHUNK = (DATA_POINTS * POINTS)

# @dataclass
# class GestureData:
#     wrist: list[float, float, float] # [x, y, z]
#     thumb_cmc: list[float, float, float] # [x, y, z]
#     thumb_mcp: list[float, float, float]
#     thumb_ip: list[float, float, float]
#     thumb_tip: list[float, float, float]
#     index_mcp: list[float, float, float]
#     index_pip: list[float, float, float]
#     index_dip: list[float, float, float]
#     index_tip: list[float, float, float]
#     middle_mcp: list[float, float, float]
#     middle_pip: list[float, float, float]
#     middle_dip: list[float, float, float]
#     middle_tip: list[float, float, float]
#     ring_mcp: list[float, float, float]
#     ring_pip: list[float, float, float]
#     ring_dip: list[float, float, float]
#     ring_tip: list[float, float, float]
#     pinky_mcp: list[float, float, float]
#     pinky_pip: list[float, float, float]
#     pinky_dip: list[float, float, float]
#     pinky_tip: list[float, float, float]

#     @classmethod
#     def from_landmark_list(cls, landmarks: list[Landmark]):
#         return cls(
#             wrist=[landmarks[0].x, landmarks[0].y, landmarks[0].z],
#             thumb_cmc=[landmarks[1].x, landmarks[1].y, landmarks[1].z],
#             thumb_mcp=[landmarks[2].x, landmarks[2].y, landmarks[2].z],
#             thumb_ip=[landmarks[3].x, landmarks[3].y, landmarks[3].z],
#             thumb_tip=[landmarks[4].x, landmarks[4].y, landmarks[4].z],
#             index_mcp=[landmarks[5].x, landmarks[5].y, landmarks[5].z],
#             index_pip=[landmarks[6].x, landmarks[6].y, landmarks[6].z],
#             index_dip=[landmarks[7].x, landmarks[7].y, landmarks[7].z],
#             index_tip=[landmarks[8].x, landmarks[8].y, landmarks[8].z],
#             middle_mcp=[landmarks[9].x, landmarks[9].y, landmarks[9].z],
#             middle_pip=[landmarks[10].x, landmarks[10].y, landmarks[10].z],
#             middle_dip=[landmarks[11].x, landmarks[11].y, landmarks[11].z],
#             middle_tip=[landmarks[12].x, landmarks[12].y, landmarks[12].z],
#             ring_mcp=[landmarks[13].x, landmarks[13].y, landmarks[13].z],
#             ring_pip=[landmarks[14].x, landmarks[14].y, landmarks[14].z],
#             ring_dip=[landmarks[15].x, landmarks[15].y, landmarks[15].z],
#             ring_tip=[landmarks[16].x, landmarks[16].y, landmarks[16].z],
#             pinky_mcp=[landmarks[17].x, landmarks[17].y, landmarks[17].z],
#             pinky_pip=[landmarks[18].x, landmarks[18].y, landmarks[18].z],
#             pinky_dip=[landmarks[19].x, landmarks[19].y, landmarks[19].z],
#             pinky_tip=[landmarks[20].x, landmarks[20].y, landmarks[20].z]
#         )

#     @classmethod
#     def from_list(cls, raw_data: list[float]):
#         return cls(
#             wrist=raw_data[0:3],
#             thumb_cmc=raw_data[3:6],
#             thumb_mcp=raw_data[6:9],
#             thumb_ip=raw_data[9:12],
#             thumb_tip=raw_data[12:15],
#             index_mcp=raw_data[15:18],
#             index_pip=raw_data[18:21],
#             index_dip=raw_data[21:24],
#             index_tip=raw_data[24:27],
#             middle_mcp=raw_data[27:30],
#             middle_pip=raw_data[30:33],
#             middle_dip=raw_data[33:36],
#             middle_tip=raw_data[36:39],
#             ring_mcp=raw_data[39:42],
#             ring_pip=raw_data[42:45],
#             ring_dip=raw_data[45:48],
#             ring_tip=raw_data[48:51],
#             pinky_mcp=raw_data[51:54],
#             pinky_pip=raw_data[54:57],
#             pinky_dip=raw_data[57:60],
#             pinky_tip=raw_data[60:63]
#         )

#     def to_list(self):
#         raw_coords = []
#         for attr in self.__dict__.values():
#             # print(attr)
#             raw_coords += attr
#         return raw_coords



# @dataclass
# class DataSample:
#     label: str
#     gestures: list[GestureData]
#     label_id: int | None = None
#     framerate: int = 30

#     @classmethod
#     def from_json(cls, json_data: dict, label_id: int = None):
#         if label_id is None:
#             label_id = json_data['label_id']
#         framerate = 30
#         if json_data.get("framerate") is not None:
#             framerate = json_data['framerate']
#         return cls(
#             label=json_data['label'],
#             label_id=label_id,
#             framerate=framerate,
#             gestures=[GestureData(**gesture) for gesture in json_data['gestures']]
#         )

#     @classmethod
#     def from_json_file(cls, file_path: str, label_id: int = None):
#         with open(file_path, 'r', encoding="utf-8") as f:
#             data = json.load(f)
#         return cls.from_json(data, label_id)

#     @classmethod
#     def from_handlandmarker(cls, hand_landmarks: HandLandmarkerResult, label, label_id):
#         return cls(
#             label=label,
#             label_id=label_id,
#             gestures=[GestureData.from_landmark_list(hand_landmarks.hand_world_landmarks[0])]
#         )

#     def to_json(self):
#         return {
#             'label': self.label,
#             'label_id': self.label_id,
#             'gestures': [gesture.__dict__ for gesture in self.gestures]
#         }

#     def to_json_file(self, file_path: str, indent: bool = False):
#         with open(file_path, 'w', encoding="utf-8") as f:
#             json.dump(self.to_json(), f, indent=indent)

#     def pushfront_gesture_from_landmarks(self, hand_landmarks: HandLandmarkerResult, allow_empty_frame: bool = True):
#         if len(hand_landmarks.hand_world_landmarks) == 0:
#             if not allow_empty_frame:
#                 raise ValueError("Empty frame not allowed")
#             else:
#                 self.gestures.insert(0, GestureData.from_list([0 for _ in range(NEURON_CHUNK)]))
#         else:
#             self.gestures.insert(0, GestureData.from_landmark_list(hand_landmarks.hand_world_landmarks[0]))

#     def samples_to_1d_array(self) -> list[float]:
#         raw_data = []
#         for gesture in self.gestures:
#             # raw_data.append(gesture.to_list())
#             raw_data += gesture.to_list()
#         return raw_data

#     def mirror_sample(self, mirror_x: bool = True, mirror_y: bool = False, mirror_z: bool = False) -> 'DataSample':
#         for i in range(len(self.gestures)):
#             for field in fields(self.gestures[i]):
#                 field_value: list[float] = getattr(self.gestures[i], field.name)
#                 if mirror_x:
#                     field_value[0] = -field_value[0]
#                 if mirror_y:
#                     field_value[1] = -field_value[1]
#                 if mirror_z:
#                     field_value[2] = -field_value[2]
#                 setattr(self.gestures[i], field.name, field_value)
#         return self

#     def rotate_sample(self, angle_x: float = 0, angle_y: float = 0, angle_z: float = 0) -> 'DataSample':
#         for i in range(len(self.gestures)):
#             for field in fields(self.gestures[i]):
#                 field_value: list[float] = getattr(self.gestures[i], field.name)
#                 field_value = rot_3d_x(field_value, angle_x)
#                 field_value = rot_3d_y(field_value, angle_y)
#                 field_value = rot_3d_z(field_value, angle_z)
#                 setattr(self.gestures[i], field.name, field_value)
#         return self

#     def randomize_points(self, factor: float = 0.005) -> 'DataSample':
#         for i in range(len(self.gestures)):
#             for field in fields(self.gestures[i]):
#                 field_value: list[float] = getattr(self.gestures[i], field.name)
#                 field_value[0] += clamp((random.random() - 0.5) * factor, -1, 1)
#                 field_value[1] += clamp((random.random() - 0.5) * factor, -1, 1)
#                 field_value[2] += clamp((random.random() - 0.5) * factor, -1, 1)
#                 setattr(self.gestures[i], field.name, field_value)
#         return self

#     def translate_hand(self, x: float = 0, y: float = 0, z: float = 0) -> 'DataSample':
#         for i in range(len(self.gestures)):
#             for field in fields(self.gestures[i]):
#                 field_value: list[float] = getattr(self.gestures[i], field.name)
#                 field_value[0] += x
#                 field_value[1] += y
#                 field_value[2] += z
#                 setattr(self.gestures[i], field.name, field_value)
#         return self

#     def deform_hand(self, x: float = 1, y: float = 1, z: float = 1) -> 'DataSample':
#         for i in range(len(self.gestures)):
#             for field in fields(self.gestures[i]):
#                 field_value: list[float] = getattr(self.gestures[i], field.name)
#                 field_value[0] *= x
#                 field_value[1] *= y
#                 field_value[2] *= z
#                 setattr(self.gestures[i], field.name, field_value)
#         return self

#     def reframe(self, target_frame: int) -> 'DataSample':
#         """Change the number of frame to execute the full gesture sequence
#         Be careful, if frame are reducedn reincresing the frame will not restore the original gesture

#         Args:
#             frame (int): Target frame number
#         """
#         def lerp(a, b, t):
#             return [a[i] + (b[i] - a[i]) * t for i in range(len(a))]

#         new_gestures: list[GestureData] = []

#         if target_frame <= 1:
#             raise ValueError("Target frame must be greater than 1")

#         for i in range(target_frame):
#             progression = i / (target_frame - 1)
#             frame_scaled_value = min(progression * (len(self.gestures) - 1), len(self.gestures) - 1)
#             # print(i, frame_scaled_value, len(self.gestures))
#             start_frame = math.floor(frame_scaled_value)
#             end_frame = math.ceil(frame_scaled_value)
#             interpolation_coef = frame_scaled_value - start_frame


#             new_gestures.append(GestureData(
#                 wrist=lerp(self.gestures[start_frame].wrist, self.gestures[end_frame].wrist, interpolation_coef),
#                 thumb_cmc=lerp(self.gestures[start_frame].thumb_cmc, self.gestures[end_frame].thumb_cmc, interpolation_coef),
#                 thumb_mcp=lerp(self.gestures[start_frame].thumb_mcp, self.gestures[end_frame].thumb_mcp, interpolation_coef),
#                 thumb_ip=lerp(self.gestures[start_frame].thumb_ip, self.gestures[end_frame].thumb_ip, interpolation_coef),
#                 thumb_tip=lerp(self.gestures[start_frame].thumb_tip, self.gestures[end_frame].thumb_tip, interpolation_coef),
#                 index_mcp=lerp(self.gestures[start_frame].index_mcp, self.gestures[end_frame].index_mcp, interpolation_coef),
#                 index_pip=lerp(self.gestures[start_frame].index_pip, self.gestures[end_frame].index_pip, interpolation_coef),
#                 index_dip=lerp(self.gestures[start_frame].index_dip, self.gestures[end_frame].index_dip, interpolation_coef),
#                 index_tip=lerp(self.gestures[start_frame].index_tip, self.gestures[end_frame].index_tip, interpolation_coef),
#                 middle_mcp=lerp(self.gestures[start_frame].middle_mcp, self.gestures[end_frame].middle_mcp, interpolation_coef),
#                 middle_pip=lerp(self.gestures[start_frame].middle_pip, self.gestures[end_frame].middle_pip, interpolation_coef),
#                 middle_dip=lerp(self.gestures[start_frame].middle_dip, self.gestures[end_frame].middle_dip, interpolation_coef),
#                 middle_tip=lerp(self.gestures[start_frame].middle_tip, self.gestures[end_frame].middle_tip, interpolation_coef),
#                 ring_mcp=lerp(self.gestures[start_frame].ring_mcp, self.gestures[end_frame].ring_mcp, interpolation_coef),
#                 ring_pip=lerp(self.gestures[start_frame].ring_pip, self.gestures[end_frame].ring_pip, interpolation_coef),
#                 ring_dip=lerp(self.gestures[start_frame].ring_dip, self.gestures[end_frame].ring_dip, interpolation_coef),
#                 ring_tip=lerp(self.gestures[start_frame].ring_tip, self.gestures[end_frame].ring_tip, interpolation_coef),
#                 pinky_mcp=lerp(self.gestures[start_frame].pinky_mcp, self.gestures[end_frame].pinky_mcp, interpolation_coef),
#                 pinky_pip=lerp(self.gestures[start_frame].pinky_pip, self.gestures[end_frame].pinky_pip, interpolation_coef),
#                 pinky_dip=lerp(self.gestures[start_frame].pinky_dip, self.gestures[end_frame].pinky_dip, interpolation_coef),
#                 pinky_tip=lerp(self.gestures[start_frame].pinky_tip, self.gestures[end_frame].pinky_tip, interpolation_coef),
#             ))

#         self.gestures = new_gestures
#         # print(self.gestures)
#         return self

#     def round_gesture_coordinates(self, decimal: int = 3) -> 'DataSample':
#         for i in range(len(self.gestures)):
#             for field in fields(self.gestures[i]):
#                 field_value: list[float] = getattr(self.gestures[i], field.name)
#                 field_value = [round(coord, decimal) for coord in field_value]
#                 setattr(self.gestures[i], field.name, field_value)
#         return self


@dataclass
class DataSample2:
    label: str
    gestures: list[DataGestures]
    framerate: int = 30
    # This attribute tells the trainset generator if the sample can be mirrored, put it to false if the gesture is not symmetrical (e.g z french sign)
    mirrorable: bool = True

    @classmethod
    def from_dict(cls, json_data: dict):
        tmp = cls(label=json_data['label'],
                  gestures=[DataGestures(**gesture) for gesture in json_data['gestures']])
        tmp.framerate = json_data.get('framerate', tmp.framerate)
        tmp.mirrorable = json_data.get('mirrorable', tmp.mirrorable)
        tmp.computeHandVelocity()
        return tmp

    @classmethod
    def from_json_file(cls, file_path: str, label_id: int = None):
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def unflat(cls, label: str, raw_data: list[float], valid_fields: list[str] = None):
        tmp = cls(label=label, gestures=[])
        len_valid_fields = (len(valid_fields) if valid_fields is not None else len(FIELDS)) * FIELD_DIMENSION
        for i in range(0, len(raw_data), len_valid_fields):
            tmp.gestures.append(DataGestures.from1DArray(raw_data[i:i + len_valid_fields], valid_fields))
        tmp.computeHandVelocity()
        return tmp

    def to_dict(self):
        tmp: dict = copy.deepcopy(self).__dict__
        tmp['gestures'] = [gesture.__dict__ for gesture in self.gestures]
        return tmp

    def to_json_file(self, file_path: str, indent: bool = False):
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    def to_tensor(self, sequence_length: int, valid_fields: list[str] = FIELDS, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        data = torch.zeros(sequence_length, len(valid_fields) * FIELD_DIMENSION, device=device)
        for i in range(min(sequence_length, len(self.gestures))):
            data[i] = self.gestures[i].toTensor(valid_fields, device)
        return data

    def flat(self, valid_fields: list[str] = None) -> list[float]:
        """Transform the gesture data into a 1D array
        WARNING: This function will not store the framerate information nor the label
        """
        raw_data: list[float] = []
        for gesture in self.gestures:
            raw_data.extend(gesture.get1DArray(valid_fields))
        return raw_data

    def insert_gesture_from_landmarks(self, position: int, hand_landmarks: HandLandmarkerResult) -> 'DataSample2':
        self.gestures.insert(position, DataGestures.buildFromHandLandmarkerResult(hand_landmarks))
        self.computeHandVelocity()
        return self

    def samples_to_1d_array(self, valid_fields: list[str] | None = None) -> list[float]:
        raw_data = []
        for gesture in self.gestures:
            raw_data.extend(gesture.get1DArray(valid_fields))
        return raw_data

    def setNonePointsRandomlyToRandomOrZero(self, proba: float = 0.1) -> 'DataSample2':
        for gest in self.gestures:
            gest.setNonePointsRandomlyToRandomOrZero(proba)
        return self

    def computeHandVelocity(self) -> 'DataSample2':
        for i in range(len(self.gestures) - 1):
            if self.gestures[i].r_hand_position is not None and self.gestures[i + 1].r_hand_position is not None:
                self.gestures[i].r_hand_velocity = [self.gestures[i].r_hand_position[k] - self.gestures[i + 1].r_hand_position[k] for k in range(3)]
            if self.gestures[i].l_hand_position is not None and self.gestures[i + 1].l_hand_position is not None:
                self.gestures[i].l_hand_velocity = [self.gestures[i].l_hand_position[k] - self.gestures[i + 1].l_hand_position[k] for k in range(3)]
        return self

    def noise_sample(self, range: float = 0.004, valid_fields: list[str] = None) -> 'DataSample2':
        """Will randomize the sample gesture points by doing `new_val = old_val + rand_val(-range, range)` to each selected point.

        Args:
            range (float, optional): Random value will be between -range and range. Defaults to 0.005.
            valid_fields (list[str], optional): Let you pick which fields should be randomized. Defaults to None (All point affected).

        Returns:
            DataSample2: Return this class instance for chaining
        """
        for gesture in self.gestures:
            gesture.noise(range, valid_fields)
        return self

    def mirror_sample(self, x: bool = True, y: bool = False, z: bool = False) -> 'DataSample2':
        for gesture in self.gestures:
            gesture.mirror(x, y, z)
        return self

    def rotate_sample(self, x: float = 0, y: float = 0, z: float = 0, valid_fields: list[str] = None) -> 'DataSample2':
        for gesture in self.gestures:
            gesture.rotate(x, y, z, valid_fields)
        return self

    def scale_sample(self, x: float = 1, y: float = 1, z: float = 1, valid_fields: list[str] = None) -> 'DataSample2':
        for gesture in self.gestures:
            gesture.scale(x, y, z, valid_fields)
        return self

    def translate_sample(self, x: float = 0, y: float = 0, z: float = 0, valid_fields: list[str] = None) -> 'DataSample2':
        for gesture in self.gestures:
            gesture.translate(x, y, z, valid_fields)
        return self

    def reframe(self, target_frame: int) -> 'DataSample2':
        """Change the number of frame to execute the full gesture sequence
        Be careful, if frame are reducedn reincresing the frame will not restore the original gesture

        Args:
            frame (int): Target frame number
        """
        if target_frame <= 1:
            raise ValueError("Target frame must be greater than 1")

        def list_lerp(a: list[float, float, float], b: list[float, float, float], t):
            if a is None or b is None:
                return None
            return [a[i] + (b[i] - a[i]) * t for i in range(len(a))]

        new_gestures: list[DataGestures] = []

        for i in range(target_frame):
            progression = i / (target_frame - 1)
            frame_scaled_value = min(progression * (len(self.gestures) - 1), len(self.gestures) - 1)
            # print(i, frame_scaled_value, len(self.gestures))
            start_frame = math.floor(frame_scaled_value)
            end_frame = math.ceil(frame_scaled_value)
            interpolation_coef = frame_scaled_value - start_frame

            new_gesture: DataGestures = DataGestures()

            for field in fields(new_gesture):
                setattr(new_gesture, field.name, list_lerp(getattr(self.gestures[start_frame], field.name), getattr(self.gestures[end_frame], field.name), interpolation_coef))

            new_gestures.append(new_gesture)

        self.gestures = new_gestures
        # print(self.gestures)
        self.computeHandVelocity()
        return self

    def set_sample_gestures_point_to(self, point_field_name: str, value: list[float]) -> 'DataSample2':
        for gesture in self.gestures:
            setattr(gesture, point_field_name, value)
        return self

    def swap_hands(self) -> 'DataSample2':
        """Should not be used.<br>
        This function is used to swap the left hand and right hand data,
        in case the hands are mirrored or the data is not in the right order.

        Returns:
            DataSample2: _description_
        """
        for gesture in self.gestures:
            gesture.swapHands()
        return self

    def move_to_one_side(self, right_side: bool = True) -> 'DataSample2':
        for gesture in self.gestures:
            gesture.moveToOneSide(right_side)
        return self
