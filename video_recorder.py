import os
import cv2
import json
import argparse
import numpy as np
import tkinter as tk
from datetime import datetime
from src.datasample import DataSample
from src.video_cropper import VideoCropper
from mediapipe.tasks.python.vision.hand_landmarker import *
from run_model import load_hand_landmarker, track_hand, draw_land_marks

ESC = 27
SPACE = 32
TAB = 9
FPS = 30

keys_index = {'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j',
              'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p', 'q': 'q', 'r': 'r', 's': 's', 't': 't',
              'u': 'u', 'v': 'v', 'w': 'w', 'x': 'x', 'y': 'y', 'z': 'z', '1': '1', '2': '2', '3': '3', '4': '4',
              '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '0': '0'}

save_folder = 'datasets/'
handland_marker = load_hand_landmarker(1)
record = cv2.VideoCapture(0)
frame_width = int(record.get(3))
frame_height = int(record.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = None
out = None

is_recording = False
is_croping = False

instructions = """Instructions:
Space: Record
Any key: Screenshot
Tab: Edit
Esc: Quit"""

parser = argparse.ArgumentParser(description="Video recording with hand detection.")
parser.add_argument("label", type=str, nargs="?", default="undefined", help="Label for the video files (default: undefined)")
args = parser.parse_args()
video_label = args.label

def create_instruction_image():
    instruction_image = np.zeros((frame_height, 300, 3), dtype=np.uint8)
    y0, dy = 50, 30
    for i, line in enumerate(instructions.split('\n')):
        y = y0 + i * dy
        cv2.putText(instruction_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return instruction_image

def update_json(json_path, file_info):
    with open(json_path, 'r') as f:
        data = json.load(f)
    data.append(file_info)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

while True:
    if not is_croping:
        ret, frame = record.read()
        if not ret:
            print("Video error.")
            break

        if is_recording:
            out.write(frame)
            result, _ = track_hand(frame, handland_marker)
            data_sample.pushfront_gesture_from_landmarks(result)
            frame = draw_land_marks(frame, result)
            cv2.putText(frame, "Recording...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        instruction_image = create_instruction_image()
        combined_frame = np.hstack((frame, instruction_image))

        cv2.imshow("Video recorder", combined_frame)

        key = cv2.waitKey(1)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if key == SPACE:
            if not is_recording:
                file_name = video_label + "_" + current_time + ".avi"
                full_save_path = save_folder + video_label + '/temp'
                data_sample = DataSample(video_label, [])

                if not os.path.exists(full_save_path):
                    os.makedirs(full_save_path)

                label_json_path = os.path.join(full_save_path, 'label.json')
                if not os.path.exists(label_json_path):
                    with open(label_json_path, 'w') as f:
                        json.dump([], f)

                output_file = os.path.join(full_save_path, file_name)
                out = cv2.VideoWriter(output_file, fourcc, FPS, (frame_width, frame_height))
                is_recording = True
            else:
                is_recording = False
                out.release()
                data_sample.to_json_file(f"{save_folder}{video_label}/{file_name}.json")
                update_json(label_json_path, {"filename": file_name, "label": video_label})

        elif key == TAB:
            is_croping = True

        elif key == ESC:
            if not is_recording:
                break
            else:
                print("Stop recording before quitting.")

        for keys in keys_index.keys():
            if key == ord(keys):
                image_name = keys_index[keys]
                full_save_path = save_folder + keys_index[keys] + '/temp'
                file_name = image_name + "_" + current_time + ".png"

                if not os.path.exists(full_save_path):
                    os.makedirs(full_save_path)

                label_json_path = os.path.join(full_save_path, 'label.json')
                if not os.path.exists(label_json_path):
                    with open(label_json_path, 'w') as f:
                        json.dump([], f)

                output_file = os.path.join(full_save_path, file_name)
                cv2.imwrite(output_file, frame)
                update_json(label_json_path, {"filename": file_name, "label": image_name})

    else:
        root = tk.Tk()
        app = VideoCropper(root)
        root.mainloop()
        is_croping = False

record.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
