import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerResult, GestureRecognizerOptions, GestureRecognizer

class HandRecognition:
    def __init__(self):
        self.target_width = 480
        self.target_height = 480

        self.base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
        self.options: GestureRecognizerOptions = vision.GestureRecognizerOptions(base_options=self.base_options,
                                          num_hands=2,
                                          min_hand_detection_confidence=0,
                                          min_hand_presence_confidence=0.1,
                                          min_tracking_confidence=0)
        self.recognizer: GestureRecognizer = vision.GestureRecognizer.create_from_options(self.options)


    def get_hand(self, image_path: str) -> GestureRecognizerResult:
        image = cv2.imread(image_path)

        original_height, original_width = image.shape[:2]

        if original_width > original_height:
            self.target_height = int(original_height / original_width * self.target_width)
        else:
            self.target_width = int(original_width / original_height * self.target_height)

        image = cv2.resize(image, (self.target_width, self.target_height))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        recognition_result = self.recognizer.recognize(mp_image)

        current_frame = image
        for hand_index, hand_landmarks in enumerate(recognition_result.hand_landmarks):
            x_min = min([landmark.x for landmark in hand_landmarks])
            x_max = max([landmark.x for landmark in hand_landmarks])
            y_min = min([landmark.y for landmark in hand_landmarks])
            y_max = max([landmark.y for landmark in hand_landmarks])

            cv2.rectangle(current_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        return current_frame
