import os
import cv2
import numpy as np
from datetime import datetime

ESC = 27
SPACE = 32

save_dir = 'datasets/source_videos'
if not os.path.exists(save_dir):
    sys.exit(f"Folder {save_dir} doesnt exist.")

record = cv2.VideoCapture(0)

frame_width = int(record.get(3))
frame_height = int(record.get(4))
fps = 15

fourcc = cv2.VideoWriter_fourcc(*'XVID') # To save in .avi
output_file = None
out = None

instructions = """Instructions:
Space: Start/Stop recording
Esc: Quit"""

is_recording = False

def create_instruction_image():
    instruction_image = np.zeros((frame_height, 300, 3), dtype=np.uint8)
    
    y0, dy = 50, 30
    for i, line in enumerate(instructions.split('\n')):
        y = y0 + i * dy
        cv2.putText(instruction_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return instruction_image

cv2.namedWindow("Video recorder", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video recorder", frame_width + 300, frame_height)

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


    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == SPACE:
        if not is_recording:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file = os.path.join(save_dir, f'output_video_{current_time}.avi')
            out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
            is_recording = True
            print("Starting to record...")
        else:
            is_recording = False
            out.release()
            print(f"Recording saved in {output_file}.")

    elif key == ESC:
        if not is_recording:
            break
        else:
            print("Stop recording before quitting.")

record.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
