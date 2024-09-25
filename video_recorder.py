import os
import cv2
import numpy as np
from datetime import datetime

ESC = 27
SPACE = 32
A_KEY = 97
fps = 30

save_dir_videos = 'datasets/source_videos'
save_dir_screenshots = 'datasets/source_images/screenshots'

if not os.path.exists(save_dir_videos):
    os.makedirs(save_dir_videos)

if not os.path.exists(save_dir_screenshots):
    os.makedirs(save_dir_screenshots)

record = cv2.VideoCapture(0)

frame_width = int(record.get(3))
frame_height = int(record.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # To save in .avi
is_recording = False
output_file = None
out = None

instructions = """Instructions:
Space: Record
A: Screenshot
Esc: Quit"""


def create_instruction_image():
    instruction_image = np.zeros((frame_height, 300, 3), dtype=np.uint8)
    
    y0, dy = 50, 30
    for i, line in enumerate(instructions.split('\n')):
        y = y0 + i * dy
        cv2.putText(instruction_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return instruction_image

while True:
    ret, frame = record.read()
    if not ret:
        print("Video error.")
        break

    if is_recording:
        out.write(frame)
        cv2.putText(frame, "Recording...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    instruction_image = create_instruction_image()
    combined_frame = np.hstack((frame, instruction_image))

    cv2.imshow("Video recorder", combined_frame)

    key = cv2.waitKey(1) & 0xFF
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if key == SPACE:
        if not is_recording:
            output_file = os.path.join(save_dir_videos, f'output_video_{current_time}.avi')
            out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
            is_recording = True
            print("Starting to record...")
        else:
            is_recording = False
            out.release()
            print(f"Recording saved in {output_file}.")
    elif key == A_KEY:
        output_file = os.path.join(save_dir_screenshots, f'screenshot_{current_time}.png')
        cv2.imwrite(output_file, frame)
        print(f"Screenshot saved in {output_file}.")

    elif key == ESC:
        if not is_recording:
            break
        else:
            print("Stop recording before quitting.")

record.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
