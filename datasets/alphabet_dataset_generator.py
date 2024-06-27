import os

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision.hand_landmarker import *
from mediapipe.tasks.python.components.containers.category import *
from mediapipe.tasks.python.components.containers.landmark import *

path_from_script = os.path.dirname(os.path.abspath(__file__))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hand_map = {
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
}

base_options = python.BaseOptions(model_asset_path=f"{path_from_script}/../hand_landmarker.task")
options: HandLandmarkerOptions = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2,
                                                              min_hand_detection_confidence=0,
                                                              min_hand_presence_confidence=0.1,
                                                              min_tracking_confidence=0)
recognizer: HandLandmarker = vision.HandLandmarker.create_from_options(options)

for folder in os.listdir(f"{path_from_script}/source_images"):
    print(f"Found file: {folder} in {path_from_script}/source_images")
    for file in os.listdir(f"{path_from_script}/source_images/{folder}"):
        print(f"Found file: {file} in {folder}")
        image = cv2.imread(f"{path_from_script}/source_images/{folder}/{file}")
        original_height, original_width = image.shape[:2]

        target_width = 1000  # Specify the desired width
        target_height = 1000  # Specify the desired height

        if original_width > original_height:
            target_height = int(original_height / original_width * target_width)
        else:
            target_width = int(original_width / original_height * target_height)

        image = cv2.resize(image, (target_width, target_height))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        recognition_result: HandLandmarkerResult = recognizer.detect(mp_image)

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
                hand_landmarks_proto.landmark.extend([
                  landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                  z=landmark.z) for landmark in hand_landmarks])
                mp_drawing.draw_landmarks(
                  current_frame,
                  hand_landmarks_proto,
                  mp_hands.HAND_CONNECTIONS,
                  mp_drawing_styles.get_default_hand_landmarks_style(),
                  mp_drawing_styles.get_default_hand_connections_style())
        if recognition_result.hand_world_landmarks:
            pass
        else:
            print(f"No hand detected for {file}")

        recognition_frame = current_frame
        original_height, original_width = recognition_frame.shape[:2]

        target_width = 192  # Specify the desired width
        target_height = 192  # Specify the desired height

        if original_width > original_height:
            target_height = int(original_height / original_width * target_width)
        else:
            target_width = int(original_width / original_height * target_height)

        recognition_frame = cv2.resize(recognition_frame, (target_width, target_height))

        if recognition_frame is not None:
            cv2.imshow('gesture_recognition', recognition_frame)
            cv2.waitKeyEx(10)

cv2.destroyAllWindows()
