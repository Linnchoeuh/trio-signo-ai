import os
import sys
from dataclasses import dataclass
import math
import copy
import random
import json

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision.hand_landmarker import *
from mediapipe.tasks.python.components.containers.category import *
from mediapipe.tasks.python.components.containers.landmark import *

current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Now you can import the module from the parent folder
from src.alphabet_recognizer import LABEL_MAP


# @dataclass
# class DataSample:
#     label: str
#     label_id: int
#     input: list[float]

def landmarks_to_list(landmarks: list[Landmark]):
    return [[landmark.x, landmark.y, landmark.z] for landmark in landmarks]

def to_1d_array(l: list[list[any]]):
    return [item for sublist in l for item in sublist]

def rot_3d_y(coords3d: list[float], angle: float):
    x, y, z = coords3d
    x_new = x * math.cos(angle) + z * math.sin(angle)
    y_new = y
    z_new = -x * math.sin(angle) + z * math.cos(angle)
    return [x_new, y_new, z_new]

def rot_3d_x(coords3d: list[float], angle: float):
    x, y, z = coords3d
    x_new = x
    y_new = y * math.cos(angle) - z * math.sin(angle)
    z_new = y * math.sin(angle) + z * math.cos(angle)
    return [x_new, y_new, z_new]

def rot_3d_z(coords3d: list[float], angle: float):
    x, y, z = coords3d
    x_new = x * math.cos(angle) - y * math.sin(angle)
    y_new = x * math.sin(angle) + y * math.cos(angle)
    z_new = z
    return [x_new, y_new, z_new]

ROTATION = math.pi / 16 # rotation amplitude
SUB_ROTATION = 32
SUB_SAMPLE = 10

path_from_script = os.path.dirname(os.path.abspath(__file__))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

dataset: list = []

base_options = python.BaseOptions(model_asset_path=f"{path_from_script}/../hand_landmarker.task")
options: HandLandmarkerOptions = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1,
                                                              min_hand_detection_confidence=0,
                                                              min_hand_presence_confidence=0.1,
                                                              min_tracking_confidence=0)
recognizer: HandLandmarker = vision.HandLandmarker.create_from_options(options)

from datasample import DataSample

for folder in os.listdir(f"{path_from_script}/source_images"):
    print(f"Found file: {folder} in {path_from_script}/source_images")
    for file in os.listdir(f"{path_from_script}/source_images/{folder}"):
        print(f"Found file: {file} in {folder}")
        image = cv2.imread(f"{path_from_script}/source_images/{folder}/{file}")
        original_height, original_width = image.shape[:2]

        target_width = 480  # Specify the desired width
        target_height = 480  # Specify the desired height

        if original_width > original_height:
            target_height = int(original_height / original_width * target_width)
        else:
            target_width = int(original_width / original_height * target_height)

        image = cv2.resize(image, (target_width, target_height))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure the image is in RGB format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        recognition_result: HandLandmarkerResult = recognizer.detect(mp_image)

        # print(recognition_result)

        current_frame = image
        for hand_index, hand_landmarks in enumerate(recognition_result.hand_landmarks):
            # Calculate the bounding box of the hand
            x_min = min([landmark.x for landmark in hand_landmarks])
            y_min = min([landmark.y for landmark in hand_landmarks])
            y_max = max([landmark.y for landmark in hand_landmarks])

            # Convert normalized coordinates to pixel values
            frame_height, frame_width = current_frame.shape[:2]
            x_min_px = int(x_min * frame_width)
            y_min_px = int(y_min * frame_height)
            y_max_px = int(y_max * frame_height)


            # Draw hand landmarks on the frame
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            mp_drawing.draw_landmarks(
                current_frame,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        if len(recognition_result.hand_world_landmarks) > 0:
            label = file[0].lower()
            sample = DataSample.from_handlandmarker(recognition_result, label=label, label_id=LABEL_MAP.label[file[0]])
            print(sample)
            if label == "0":
                label = "_null"
            target_folder = f"{path_from_script}/tmp/{label}"
            os.makedirs(target_folder, exist_ok=True)

            i = 0
            file_name = f"{label}_{i}.json"
            files = os.listdir(target_folder)
            while file_name in files:
                i += 1
                file_name = f"{label}_{i}.json"

            print(f"Writing to {file_name}")
            with open(f"{target_folder}/{file_name}", "w", encoding='utf-8') as f:
                f.write(json.dumps(sample.to_json(), indent=4, ensure_ascii=False))

        for landmarks in recognition_result.hand_world_landmarks:
            # Add the original hand
            print(landmarks)
            # original = landmarks_to_list(landmarks)
            # dataset.append(DataSample(label=file[0], label_id=LABEL_MAP.label[file[0]], input=to_1d_array(original)))
            """

            # Add the mirrored hand
            original_mirror = copy.deepcopy(original)
            for i in range(len(original_mirror)):
                original_mirror[i][0] = -original_mirror[i][0]
            dataset.append(DataSample(label=file[0], label_id=LABEL_MAP.label[file[0]], input=to_1d_array(original_mirror)))

            # Create variation of the original hand
            for origin in [original, original_mirror]:
                for _ in range(SUB_SAMPLE):
                    original_random = copy.deepcopy(origin)
                    for i in range(len(original_random)):
                        original_random[i][0] += (random.random() - 0.5) * 0.001
                        original_random[i][1] += (random.random() - 0.5) * 0.001
                        original_random[i][2] += (random.random() - 0.5) * 0.001
                    dataset.append(DataSample(label=file[0], label_id=LABEL_MAP.label[file[0]], input=to_1d_array(original_random)))

            # Add the rotated hands equivalent to the original and mirrored hands
            for j in range(SUB_ROTATION):
                for origin in [original, original_mirror]:
                    rotated = copy.deepcopy(original)
                    rotx = random.randint(int(-SUB_ROTATION / 2), int(SUB_ROTATION / 2))
                    roty = random.randint(int(-SUB_ROTATION / 2), int(SUB_ROTATION / 2))
                    rotz = random.randint(int(-SUB_ROTATION / 2), int(SUB_ROTATION / 2))
                    for i in range(len(rotated)):
                        rotated[i] = rot_3d_y(rotated[i], ROTATION + rotx * ((ROTATION / SUB_ROTATION)))
                        rotated[i] = rot_3d_x(rotated[i], ROTATION + roty * ((ROTATION / SUB_ROTATION)))
                        rotated[i] = rot_3d_z(rotated[i], ROTATION + rotz * ((ROTATION / SUB_ROTATION)))
                    dataset.append(DataSample(label=file[0], label_id=LABEL_MAP.label[file[0]], input=to_1d_array(rotated)))

                    # Create variation of the rotated hand
                    for _ in range(SUB_SAMPLE):
                        rotated_random = copy.deepcopy(rotated)
                        for i in range(len(rotated_random)):
                            rotated_random[i][0] += (random.random() - 0.5) * 0.001
                            rotated_random[i][1] += (random.random() - 0.5) * 0.001
                            rotated_random[i][2] += (random.random() - 0.5) * 0.001
                        dataset.append(DataSample(label=file[0], label_id=LABEL_MAP.label[file[0]], input=to_1d_array(rotated_random)))

        """

        if not recognition_result.hand_world_landmarks:
            print(f"No hand detected for {file}")

        recognition_frame = current_frame

        if recognition_frame is not None:
            cv2.imshow('gesture_recognition', recognition_frame)
            cv2.waitKeyEx(1)

cv2.destroyAllWindows()

print(f"Dataset size: {len(dataset)}")

with open(f"{path_from_script}/../alphabet_dataset.json", "w", encoding='utf-8') as f:
    f.write(json.dumps([{"label": sample.label, "label_id": sample.label_id, "input": sample.input} for sample in dataset], indent=4, ensure_ascii=False))
