import math
import random
from dataclasses import dataclass, fields
from typing import Generic, TypeVar

from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision.hand_landmarker import *
from mediapipe.tasks.python.components.containers.category import *
from mediapipe.tasks.python.components.containers.landmark import *

from src.rot_3d import *
from src.tools import *


def is_valid_field(field_name: str, valid_fields: list[str] | None) -> bool:
    return valid_fields is None or field_name in valid_fields

T = TypeVar('T')
@dataclass
class Gestures(Generic[T]):
    # NEVER CHANGE THE POINTS ORDER OR IT WILL BACKWARD COMPATIBILITY

    # Left hand data
    l_hand_position: T = None
    l_wrist: T = None
    l_thumb_cmc: T = None
    l_thumb_mcp: T = None
    l_thumb_ip: T = None
    l_thumb_tip: T = None
    l_index_mcp: T = None
    l_index_pip: T = None
    l_index_dip: T = None
    l_index_tip: T = None
    l_middle_mcp: T = None
    l_middle_pip: T = None
    l_middle_dip: T = None
    l_middle_tip: T = None
    l_ring_mcp: T = None
    l_ring_pip: T = None
    l_ring_dip: T = None
    l_ring_tip: T = None
    l_pinky_mcp: T = None
    l_pinky_pip: T = None
    l_pinky_dip: T = None
    l_pinky_tip: T = None

    # Right hand data
    r_hand_position: T = None
    r_wrist: T = None
    r_thumb_cmc: T = None
    r_thumb_mcp: T = None
    r_thumb_ip: T = None
    r_thumb_tip: T = None
    r_index_mcp: T = None
    r_index_pip: T = None
    r_index_dip: T = None
    r_index_tip: T = None
    r_middle_mcp: T = None
    r_middle_pip: T = None
    r_middle_dip: T = None
    r_middle_tip: T = None
    r_ring_mcp: T = None
    r_ring_pip: T = None
    r_ring_dip: T = None
    r_ring_tip: T = None
    r_pinky_mcp: T = None
    r_pinky_pip: T = None
    r_pinky_dip: T = None
    r_pinky_tip: T = None


@dataclass
class ActiveGestures(Gestures[bool | None]):
    """ActiveGestures class defines what the model will take into account when predicting gestures.
    For example, if the model is only interested in the position of the hands, then only the hand_position fields will be set to True,
    the rest such as the fingers positions will be set to False and will be ignored by the model.

    Args:
        Gestures (_type_): _description_
    """
    @classmethod
    def buildWithPreset(self, gestures_to_set: Gestures[bool | None] | list[Gestures[bool | None]]) -> "ActiveGestures":
        tmp = ActiveGestures()
        tmp.setActiveGestures(gestures_to_set)
        return tmp

    def setActiveGestures(self, gestures_to_set: Gestures[bool | None] | list[Gestures[bool | None]]) -> None:
        if not isinstance(gestures_to_set, list):
            gestures_to_set = [gestures_to_set]

        self.resetActiveGestures()

        for gesture in gestures_to_set:
            for field in fields(gesture):
                field_data = getattr(gesture, field.name)
                if field_data is not None:
                    setattr(self, field.name, field_data)

    def resetActiveGestures(self) -> None:
        for field in fields(self):
            setattr(self, field.name, None)

    def activateAllGesture(self) -> None:
        for field in fields(self):
            setattr(self, field.name, True)

    def deactivateAllGesture(self) -> None:
        for field in fields(self):
            setattr(self, field.name, False)

    def getActiveFields(self) -> list[str]:
        active_fields = []
        for field in fields(self):
            if getattr(self, field.name):
                active_fields.append(field.name)
        return active_fields

LEFT_HAND_POINTS = ActiveGestures(
    l_wrist=True,
    l_thumb_cmc=True,
    l_thumb_mcp=True,
    l_thumb_ip=True,
    l_thumb_tip=True,
    l_index_mcp=True,
    l_index_pip=True,
    l_index_dip=True,
    l_index_tip=True,
    l_middle_mcp=True,
    l_middle_pip=True,
    l_middle_dip=True,
    l_middle_tip=True,
    l_ring_mcp=True,
    l_ring_pip=True,
    l_ring_dip=True,
    l_ring_tip=True,
    l_pinky_mcp=True,
    l_pinky_pip=True,
    l_pinky_dip=True,
    l_pinky_tip=True,
)
LEFT_HAND_POSITION = ActiveGestures(
    l_hand_position=True
)
LEFT_HAND_FULL = ActiveGestures.buildWithPreset([LEFT_HAND_POINTS, LEFT_HAND_POSITION])
RIGHT_HAND_POINTS = ActiveGestures(
    r_wrist=True,
    r_thumb_cmc=True,
    r_thumb_mcp=True,
    r_thumb_ip=True,
    r_thumb_tip=True,
    r_index_mcp=True,
    r_index_pip=True,
    r_index_dip=True,
    r_index_tip=True,
    r_middle_mcp=True,
    r_middle_pip=True,
    r_middle_dip=True,
    r_middle_tip=True,
    r_ring_mcp=True,
    r_ring_pip=True,
    r_ring_dip=True,
    r_ring_tip=True,
    r_pinky_mcp=True,
    r_pinky_pip=True,
    r_pinky_dip=True,
    r_pinky_tip=True,
)
RIGHT_HAND_POSITION = ActiveGestures(
    r_hand_position=True
)
RIGHT_HAND_FULL = ActiveGestures.buildWithPreset([RIGHT_HAND_POINTS, RIGHT_HAND_POSITION])
HANDS_POINTS = ActiveGestures.buildWithPreset([LEFT_HAND_POINTS, RIGHT_HAND_POINTS])
HANDS_POSITION = ActiveGestures.buildWithPreset([LEFT_HAND_POSITION, RIGHT_HAND_POSITION])

HANDS_FULL = ActiveGestures.buildWithPreset([LEFT_HAND_FULL, RIGHT_HAND_FULL])

ACTIVATED_GESTURES_PRESETS: dict[str, tuple[ActiveGestures, str]] = {
    'left_hand_points': (
        LEFT_HAND_POINTS,
        "Will only provide information about left hand finger position or hand rotation."
    ),
    'left_hand_position': (
        LEFT_HAND_POSITION,
        "Will only provide information about left hand position."
    ),
    'left_hand_full': (
        LEFT_HAND_FULL,
        "Will provide information about left hand finger position, hand rotation and position."
    ),
    'right_hand_points': (
        RIGHT_HAND_POINTS,
        "Will only provide information about right hand finger position or hand rotation."
    ),
    'right_hand_position': (
        RIGHT_HAND_POSITION,
        "Will only provide information about right hand position."
    ),
    'right_hand_full': (
        RIGHT_HAND_FULL,
        "Will provide information about right hand finger position, hand rotation and position."
    ),
    'hands_points': (
        HANDS_POINTS,
        "Will only provide information about both hands finger position and hands rotation."
    ),
    'hands_position': (
        HANDS_POSITION,
        "Will only provide information about both hands position."
    ),
    'hands_full': (
        HANDS_FULL,
        "Will provide information about both hands finger position, hands rotation and position."
    )
}

CACHE_HANDS_POINTS: list[str] = HANDS_POINTS.getActiveFields()
CACHE_HANDS_POSITION: list[str] = HANDS_POSITION.getActiveFields()


@dataclass
class DataGestures(Gestures[list[float, float, float] | None]):

    @classmethod
    def buildFromHandLandmarkerResult(self, landmark_result: HandLandmarkerResult) -> "DataGestures":
        tmp = DataGestures()
        tmp.setHandsFromHandLandmarkerResult(landmark_result)
        return tmp

    def setHandsFromHandLandmarkerResult(self, landmark_result: HandLandmarkerResult) -> None:
        """Convert the HandLandmarkerResult object into a DataGestures object.

        HandLandmarkerResult.hand_landmark represent the position of the hand in the image.
        HandLandmarkerResult.hand_world_landmarks represent a normalized hand that is not altered by the position or distance of the camera.

        Args:
            landmark_result (HandLandmarkerResult): _description_
        """
        for i in range(len(landmark_result.hand_world_landmarks)):
            handlandmark: list[NormalizedLandmark] = landmark_result.hand_landmarks[i]
            handworldlandmark: list[NormalizedLandmark] = landmark_result.hand_world_landmarks[i]
            if landmark_result.handedness[i][0].category_name == "Right":
                """
                We use the wrist position to get the hand location
                then we substract 0.5 to center the hand since
                handlandmark elements store their position in a range of 0 to 1.
                Doing so will ease operation such as mirroring or rotation.
                """
                self.r_hand_position = [handlandmark[0].x - 0.5, handlandmark[0].y - 0.5, handlandmark[0].z - 0.5]

                # Adding position of each finger articulation
                self.r_wrist=[handworldlandmark[0].x, handworldlandmark[0].y, handworldlandmark[0].z],
                self.r_thumb_cmc=[handworldlandmark[1].x, handworldlandmark[1].y, handworldlandmark[1].z],
                self.r_thumb_mcp=[handworldlandmark[2].x, handworldlandmark[2].y, handworldlandmark[2].z],
                self.r_thumb_ip=[handworldlandmark[3].x, handworldlandmark[3].y, handworldlandmark[3].z],
                self.r_thumb_tip=[handworldlandmark[4].x, handworldlandmark[4].y, handworldlandmark[4].z],
                self.r_index_mcp=[handworldlandmark[5].x, handworldlandmark[5].y, handworldlandmark[5].z],
                self.r_index_pip=[handworldlandmark[6].x, handworldlandmark[6].y, handworldlandmark[6].z],
                self.r_index_dip=[handworldlandmark[7].x, handworldlandmark[7].y, handworldlandmark[7].z],
                self.r_index_tip=[handworldlandmark[8].x, handworldlandmark[8].y, handworldlandmark[8].z],
                self.r_middle_mcp=[handworldlandmark[9].x, handworldlandmark[9].y, handworldlandmark[9].z],
                self.r_middle_pip=[handworldlandmark[10].x, handworldlandmark[10].y, handworldlandmark[10].z],
                self.r_middle_dip=[handworldlandmark[11].x, handworldlandmark[11].y, handworldlandmark[11].z],
                self.r_middle_tip=[handworldlandmark[12].x, handworldlandmark[12].y, handworldlandmark[12].z],
                self.r_ring_mcp=[handworldlandmark[13].x, handworldlandmark[13].y, handworldlandmark[13].z],
                self.r_ring_pip=[handworldlandmark[14].x, handworldlandmark[14].y, handworldlandmark[14].z],
                self.r_ring_dip=[handworldlandmark[15].x, handworldlandmark[15].y, handworldlandmark[15].z],
                self.r_ring_tip=[handworldlandmark[16].x, handworldlandmark[16].y, handworldlandmark[16].z],
                self.r_pinky_mcp=[handworldlandmark[17].x, handworldlandmark[17].y, handworldlandmark[17].z],
                self.r_pinky_pip=[handworldlandmark[18].x, handworldlandmark[18].y, handworldlandmark[18].z],
                self.r_pinky_dip=[handworldlandmark[19].x, handworldlandmark[19].y, handworldlandmark[19].z],
                self.r_pinky_tip=[handworldlandmark[20].x, handworldlandmark[20].y, handworldlandmark[20].z]
            else:
                """
                We use the wrist position to get the hand location
                then we substract 0.5 to center the hand since
                handlandmark elements store their position in a range of 0 to 1.
                Doing so will ease operation such as mirroring or rotation.
                """
                self.l_hand_position = [handlandmark[0].x, handlandmark[0].y, handlandmark[0].z]

                # Adding position of each finger articulation
                self.l_wrist=[handworldlandmark[0].x, handworldlandmark[0].y, handworldlandmark[0].z],
                self.l_thumb_cmc=[handworldlandmark[1].x, handworldlandmark[1].y, handworldlandmark[1].z],
                self.l_thumb_mcp=[handworldlandmark[2].x, handworldlandmark[2].y, handworldlandmark[2].z],
                self.l_thumb_ip=[handworldlandmark[3].x, handworldlandmark[3].y, handworldlandmark[3].z],
                self.l_thumb_tip=[handworldlandmark[4].x, handworldlandmark[4].y, handworldlandmark[4].z],
                self.l_index_mcp=[handworldlandmark[5].x, handworldlandmark[5].y, handworldlandmark[5].z],
                self.l_index_pip=[handworldlandmark[6].x, handworldlandmark[6].y, handworldlandmark[6].z],
                self.l_index_dip=[handworldlandmark[7].x, handworldlandmark[7].y, handworldlandmark[7].z],
                self.l_index_tip=[handworldlandmark[8].x, handworldlandmark[8].y, handworldlandmark[8].z],
                self.l_middle_mcp=[handworldlandmark[9].x, handworldlandmark[9].y, handworldlandmark[9].z],
                self.l_middle_pip=[handworldlandmark[10].x, handworldlandmark[10].y, handworldlandmark[10].z],
                self.l_middle_dip=[handworldlandmark[11].x, handworldlandmark[11].y, handworldlandmark[11].z],
                self.l_middle_tip=[handworldlandmark[12].x, handworldlandmark[12].y, handworldlandmark[12].z],
                self.l_ring_mcp=[handworldlandmark[13].x, handworldlandmark[13].y, handworldlandmark[13].z],
                self.l_ring_pip=[handworldlandmark[14].x, handworldlandmark[14].y, handworldlandmark[14].z],
                self.l_ring_dip=[handworldlandmark[15].x, handworldlandmark[15].y, handworldlandmark[15].z],
                self.l_ring_tip=[handworldlandmark[16].x, handworldlandmark[16].y, handworldlandmark[16].z],
                self.l_pinky_mcp=[handworldlandmark[17].x, handworldlandmark[17].y, handworldlandmark[17].z],
                self.l_pinky_pip=[handworldlandmark[18].x, handworldlandmark[18].y, handworldlandmark[18].z],
                self.l_pinky_dip=[handworldlandmark[19].x, handworldlandmark[19].y, handworldlandmark[19].z],
                self.l_pinky_tip=[handworldlandmark[20].x, handworldlandmark[20].y, handworldlandmark[20].z]
        return self

    def setPointToZero(self, point: str) -> "DataGestures":
        setattr(self, point, [0, 0, 0])
        return self

    def setPointToRandom(self, point: str) -> "DataGestures":
        if point in CACHE_HANDS_POSITION:
            setattr(self, point, [rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1)])
        elif point in CACHE_HANDS_POINTS:
            # 0.15 is the max value I can find on hand landmark
            setattr(self, point, [rand_fix_interval(0.15), rand_fix_interval(0.15), rand_fix_interval(0.15)])
        return self

    def setAllPointsToZero(self) -> "DataGestures":
        for field in fields(self):
            self.setPointToZero(field.name)
        return self

    def setAllPointsToRandom(self) -> "DataGestures":
        for field in fields(self):
            self.setPointToRandom(field.name)
        return self

    def setNonePointsToZero(self) -> "DataGestures":
        for field in fields(self):
            if getattr(self, field.name) is None:
                self.setPointToZero(field.name)
        return self

    def setNonePointsToRandom(self) -> "DataGestures":
        for field in fields(self):
            if getattr(self, field.name) is None:
                self.setPointToRandom(field.name)
        return self

    def setNonePointsRandomlyToRandomOrZero(self, proba: float = 0.1) -> "DataGestures":
        for field in fields(self):
            if getattr(self, field.name) is None:
                if random.random() < proba:
                    self.setPointToZero(field.name)
                else:
                    self.setPointToRandom(field.name)
        return self

    def get1DArray(self, active_gestures: ActiveGestures | None = None) -> list[float]:
        data = []
        for field in fields(self):
            if active_gestures is None or getattr(active_gestures, field.name):
                attr: list[float, float, float] | None = getattr(self, field.name)
                if attr is None:
                    attr = [0, 0, 0]
                    # raise ValueError(f"Field {field.name} is None")
                data.extend(attr)
        return data

    def noise(self, range: float = 0.005, valid_fields: list[str] | None = None) -> "DataGestures":
        """Will randomize the gesture points by doing `new_val = old_val + rand_val(-range, range)` to each selected point.

        Args:
            range (float, optional): Random value will be between -range and range. Defaults to 0.005.
            valid_fields (list[str], optional): Let you pick which fields should be randomized. Defaults to None (All point affected).

        Returns:
            DataSample2: Return this class instance for chaining
        """
        for field in fields(self):
            if not is_valid_field(field.name, valid_fields):
                continue
            field_value: list[float] = getattr(self, field.name)
            if field_value is not None:
                field_value[0] += rand_fix_interval(range)
                field_value[1] += rand_fix_interval(range)
                field_value[2] += rand_fix_interval(range)
                setattr(self, field.name, field_value)
        return self

    def mirror(self, x: bool = True, y: bool = False, z: bool = False) -> "DataGestures":
        for field in fields(self):
            field_value: list[float] = getattr(self, field.name)
            if field_value is None:
                continue
            if x:
                field_value[0] *= -1
            if y:
                field_value[1] *= -1
            if z:
                field_value[2] *= -1
            setattr(self, field.name, field_value)

        # Mirroring the hand make the hand become the opposite hand
        # This if statement will swap the left hand and right hand data
        if (x + y + z) % 2 == 1:
            self.r_wrist, self.l_wrist = self.l_wrist, self.r_wrist

            self.r_thumb_cmc, self.l_thumb_cmc = self.l_thumb_cmc, self.r_thumb_cmc
            self.r_thumb_mcp, self.l_thumb_mcp = self.l_thumb_mcp, self.r_thumb_mcp
            self.r_thumb_ip, self.l_thumb_ip = self.l_thumb_ip, self.r_thumb_ip
            self.r_thumb_tip, self.l_thumb_tip = self.l_thumb_tip, self.r_thumb_tip

            self.r_index_mcp, self.l_index_mcp = self.l_index_mcp, self.r_index_mcp
            self.r_index_pip, self.l_index_pip = self.l_index_pip, self.r_index_pip
            self.r_index_dip, self.l_index_dip = self.l_index_dip, self.r_index_dip
            self.r_index_tip, self.l_index_tip = self.l_index_tip, self.r_index_tip

            self.r_middle_mcp, self.l_middle_mcp = self.l_middle_mcp, self.r_middle_mcp
            self.r_middle_pip, self.l_middle_pip = self.l_middle_pip, self.r_middle_pip
            self.r_middle_dip, self.l_middle_dip = self.l_middle_dip, self.r_middle_dip
            self.r_middle_tip, self.l_middle_tip = self.l_middle_tip, self.r_middle_tip

            self.r_ring_mcp, self.l_ring_mcp = self.l_ring_mcp, self.r_ring_mcp
            self.r_ring_pip, self.l_ring_pip = self.l_ring_pip, self.r_ring_pip
            self.r_ring_dip, self.l_ring_dip = self.l_ring_dip, self.r_ring_dip
            self.r_ring_tip, self.l_ring_tip = self.l_ring_tip, self.r_ring_tip

            self.r_pinky_mcp, self.l_pinky_mcp = self.l_pinky_mcp, self.r_pinky_mcp
            self.r_pinky_pip, self.l_pinky_pip = self.l_pinky_pip, self.r_pinky_pip
            self.r_pinky_dip, self.l_pinky_dip = self.l_pinky_dip, self.r_pinky_dip
            self.r_pinky_tip, self.l_pinky_tip = self.l_pinky_tip, self.r_pinky_tip
        return self

    def rotate(self, x: float = 0, y: float = 0, z: float = 0, valid_fields: list[str] | None = None) -> "DataGestures":
        for field in fields(self):
            if not is_valid_field(field.name, valid_fields):
                continue
            field_value: list[float] = getattr(self, field.name)
            if field_value is None:
                continue
            field_value = rot_3d_x(field_value, x)
            field_value = rot_3d_y(field_value, y)
            field_value = rot_3d_z(field_value, z)
            setattr(self, field.name, field_value)
        return self

    def scale(self, x: float = 1, y: float = 1, z: float = 1, valid_fields: list[str] | None = None) -> "DataGestures":
        for field in fields(self):
            if not is_valid_field(field.name, valid_fields):
                print(f"Skipping {field.name}")
                continue
            field_value: list[float] = getattr(self, field.name)
            if field_value is None:
                continue
            field_value[0] *= x
            field_value[1] *= y
            field_value[2] *= z
            setattr(self, field.name, field_value)
        return self

    def translate(self, x: float = 0, y: float = 0, z: float = 0, valid_fields: list[str] | None = None) -> "DataGestures":
        for field in fields(self):
            if not is_valid_field(field.name, valid_fields):
                continue
            field_value: list[float] = getattr(self, field.name)
            if field_value is None:
                continue
            field_value[0] += x
            field_value[1] += y
            field_value[2] += z
            setattr(self, field.name, field_value)
        return self
