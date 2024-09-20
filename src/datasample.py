from dataclasses import dataclass
from mediapipe.tasks.python.vision.hand_landmarker import *
from mediapipe.tasks.python.components.containers.landmark import *

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

@dataclass
class DataSample:
    label: str
    label_id: int
    gestures: list[GestureData]

    @classmethod
    def from_json(cls, json_data):
        return cls(
            label=json_data['label'],
            label_id=json_data['label_id'],
            gestures=[GestureData(**gesture) for gesture in json_data['gestures']]
        )

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
