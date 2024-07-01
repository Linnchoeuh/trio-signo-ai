import os
import time

import cv2
from flask import Flask, request, jsonify

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision.hand_landmarker import *
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

from src.alphabet_recognizer import *


def get_alpahabet():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file.save(file.filename)
        image = cv2.imread(file.filename)
        os.remove(file.filename)

        original_height, original_width = image.shape[:2]

        target_width = 480  # Specify the desired width
        target_height = 480  # Specify the desired height

        if original_width > original_height:
            target_height = int(original_height / original_width * target_width)
        else:
            target_width = int(original_width / original_height * target_height)

        image = cv2.resize(image, (target_width, target_height))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure the image is in RGB format

        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options: HandLandmarkerOptions = vision.HandLandmarkerOptions(base_options=base_options,
                                                                      num_hands=1,
                                                                      min_hand_detection_confidence=0,
                                                                      min_hand_presence_confidence=0.1,
                                                                      min_tracking_confidence=0)
        recognizer: HandLandmarker = vision.HandLandmarker.create_from_options(options)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        recognition_result: HandLandmarkerResult = recognizer.detect(mp_image)
        print(recognition_result)
        # print(recognition_result)

        current_frame = image
        for hand_index, hand_landmarks in enumerate(recognition_result.hand_landmarks):

            # Draw hand landmarks on the frame
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                              z=landmark.z) for landmark in
              hand_landmarks
            ])
            mp_drawing.draw_landmarks(
              current_frame,
              hand_landmarks_proto,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())

        recognition_frame = current_frame

        if len(recognition_result.hand_world_landmarks) < 1:
            return jsonify({'message': None}), 200

        alphabet_model = LSFAlphabetRecognizer()
        alphabet_model.load_state_dict(torch.load('model.pth'))


        # cv2.imshow('gesture_recognition', recognition_frame)
        # cv2.waitKeyEx(1000)
        # cv2.destroyAllWindows()
        return jsonify({'message': LABEL_MAP.id[alphabet_model.use(LandmarksTo1DArray(recognition_result.hand_world_landmarks[0]))]}), 200
