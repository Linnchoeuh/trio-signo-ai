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

        # Label box parameters
        label_text_color = (255, 255, 255)  # white
        label_font_size = 1
        label_thickness = 2

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

        if recognition_frame is not None:
            cv2.imshow('gesture_recognition', recognition_frame)
            cv2.waitKeyEx(1000)
            cv2.destroyAllWindows()
        return jsonify({'message': 'File successfully uploaded'}), 200
