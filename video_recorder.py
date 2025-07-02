import os
import cv2
import json
import argparse
import numpy as np
import copy
from datetime import datetime
from src.datasample import DataSample
from src.run_model import load_hand_landmarker, track_hand, draw_land_marks, recognize_sign
from src.model_class.sign_recognizer_v1 import *
from src.model_class.transformer_sign_recognizer import *
from src.draw_gestures import draw_gestures

from src.video_recorder.face_detection import track_face
from src.video_recorder.body_detection import track_body
from src.video_recorder.cv_drawer import CVDrawer

ESC = 27
SPACE = 32
TAB = 9
FPS = 30

keys_index = {'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j',
              'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p', 'q': 'q', 'r': 'r', 's': 's', 't': 't',
              'u': 'u', 'v': 'v', 'w': 'w', 'x': 'x', 'y': 'y', 'z': 'z', '1': '1', '2': '2', '3': '3', '4': '4',
              '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '0': '_null'}


parser = argparse.ArgumentParser(
    description="Sign recognition with video recording.")
parser.add_argument("--label", type=str, nargs="?", default="undefined",
                    help="Label for the video files (default: undefined)")
parser.add_argument("--model", required=True,
                    help="Path to the folder containing the sign recognition model.")
parser.add_argument("--counter-example", action='store_true',
                    help="Will save the sign as a counter example of the label set in --label.")
parser.add_argument("--face", action='store_true',
                    help="Enable face tracking.")
parser.add_argument("--body", action='store_true',
                    help="Enable body tracking.")
parser.add_argument("--screenshot-delay", type=float, default=0,
                    help="Delay in seconds before taking a screenshot after pressing a key. Default is 0 (no delay).")

args = parser.parse_args()
SAVE_FOLDER: str = "datasets"
MEDIA_SAVE_FOLDER: str = "media_datasets"
SUB_FOLDER = "valid" if not args.counter_example else "counter_examples"

video_label = args.label
counter_example = args.counter_example
screenshot_delay = args.screenshot_delay

print("Loading sign recognition model...")
sign_rec: SignRecognizerTransformer = SignRecognizerTransformer.loadModelFromDir(
    args.model)

print("Loading hand landmarker...")
handland_marker = load_hand_landmarker(2)


def list_available_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


available_cameras = list_available_cameras()
print(available_cameras)

record: cv2.VideoCapture = cv2.VideoCapture(available_cameras[0])
frame_width = int(record.get(3))
frame_height = int(record.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

is_recording = False
is_croping = False
remaining_delay = 0
countdown_active = False

frame_history: DataSample = DataSample("", [])
prev_sign = -1
prev_display = -1

instructions = """Instructions:
Space: Record
Any key: Screenshot
Tab: Edit
Esc: Quit"""


def create_instruction_image():
    instruction_image = np.zeros((frame_height, 300, 3), dtype=np.uint8)
    y0, dy = 50, 30
    for i, line in enumerate(instructions.split('\n')):
        y = y0 + i * dy
        cv2.putText(instruction_image, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return instruction_image


def update_json(json_path, file_info):
    with open(json_path, 'r') as f:
        data = json.load(f)
    data.append(file_info)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

cv_drawer = CVDrawer(frame_width, frame_height)

while True:
    if not is_croping:
        ret, frame = record.read()
        if not ret:
            print("Video error.")
            break

        og_frame = cv2.flip(frame, 1)
        frame = copy.deepcopy(og_frame)

        result, _ = track_hand(frame, handland_marker)

        face_result = None
        if args.face:
            frame, face_result = track_face(frame)

        body_result = None
        if args.body:
            frame, body_result = track_body(frame)

        # frame = draw_land_marks(frame, result)
        frame_history.insertGestureFromLandmarks(
            0, result, face_result, body_result)

        cv_drawer.update_frame(frame)
        draw_gestures(frame_history.gestures[0], cv_drawer.draw_line,
                      cv_drawer.draw_point)

        while len(frame_history.gestures) > sign_rec.info.memory_frame:
            frame_history.gestures.pop(-1)

        if sign_rec.info.one_side:
            frame_history.move_to_one_side()

        recognized_sign, sign_rec_time = recognize_sign(
            frame_history, sign_rec, sign_rec.info.active_gestures.getActiveFields()
        )

        text = "undefined"
        if prev_sign != recognized_sign:
            prev_display = prev_sign
            prev_sign = recognized_sign
        if recognized_sign != -1:
            text = f"{sign_rec.info.labels[recognized_sign]} prev({
                sign_rec.info.labels[prev_display]})"

        cv2.putText(frame, text, (49, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.01, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        if is_recording:
            out.write(og_frame)
            data_sample.insertGestureFromLandmarks(0, result)
            cv2.putText(frame, "Recording...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        instruction_image = create_instruction_image()
        combined_frame = np.hstack((frame, instruction_image))
        cv2.imshow("Video recorder", combined_frame)

        key = cv2.waitKey(1)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if key == SPACE:
            if not is_recording:
                file_name = video_label + "_" + current_time + ".avi"
                full_save_path = os.path.join(MEDIA_SAVE_FOLDER, video_label, SUB_FOLDER)
                data_sample = DataSample(video_label, [])

                os.makedirs(full_save_path, exist_ok=True)
                label_json_path = os.path.join(full_save_path, 'label.json')
                if not os.path.exists(label_json_path):
                    with open(label_json_path, 'w') as f:
                        json.dump([], f)

                output_file = os.path.join(full_save_path, file_name)
                out = cv2.VideoWriter(
                    output_file, fourcc, FPS, (frame_width, frame_height))
                is_recording = True
            else:
                is_recording = False
                out.release()
                tmp_path: str = os.path.join(SAVE_FOLDER, video_label, SUB_FOLDER)
                os.makedirs(tmp_path, exist_ok=True)
                data_sample.toJsonFile(os.path.join(tmp_path ,f"{file_name}.json"))
                update_json(label_json_path, {
                            "filename": file_name, "label": video_label})

        elif key == TAB:
            is_croping = True

        elif key == ESC:
            if not is_recording:
                break
            else:
                print("Stop recording before quitting.")

        for keys in keys_index.keys():
            if key == ord(keys) and not countdown_active:
                countdown_active = True
                remaining_delay = screenshot_delay

                image_label = keys_index[keys]
                file_name = image_label + "_" + current_time + ".png"
                full_save_path = os.path.join(MEDIA_SAVE_FOLDER, image_label, SUB_FOLDER)

                os.makedirs(full_save_path, exist_ok=True)
                label_json_path = os.path.join(full_save_path, 'label.json')
                if not os.path.exists(label_json_path):
                    with open(label_json_path, 'w') as f:
                        json.dump([], f)

                output_file = os.path.join(full_save_path, file_name)
                image_sample = DataSample(image_label, [])

        if countdown_active:
            remaining_delay -= 1 / FPS

            if remaining_delay <= 0:
                countdown_active = False
                cv2.imwrite(output_file, og_frame)
                update_json(label_json_path, {"filename": file_name, "label": image_label})

                result, _ = track_hand(og_frame, handland_marker)
                image_sample.insertGestureFromLandmarks(
                    0, result, face_result, body_result)
                tmp_path: str = os.path.join(SAVE_FOLDER, image_label, SUB_FOLDER)
                os.makedirs(tmp_path, exist_ok=True)
                image_sample.toJsonFile(os.path.join(tmp_path, f"{file_name}.json"))

record.release()
if out:
    out.release()
cv2.destroyAllWindows()
