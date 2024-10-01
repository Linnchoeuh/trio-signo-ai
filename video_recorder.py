import os
import cv2
import json
import numpy as np
import tkinter as tk
from datetime import datetime
from tkinter import simpledialog
from src.video_cropper import VideoCropper

ESC = 27
SPACE = 32
A_KEY = 97
B_KEY = 98
FPS = 30

keys_index = {'a' : 'a',
        'b' : 'b',
        'c' : 'c',
        'd' : 'd',
        'e' : 'e',
        'f' : 'f',
        'g' : 'g',
        'h' : 'h',
        'i' : 'i',
        'j' : 'j',
        'k' : 'k',
        'l' : 'l',
        'm' : 'm',
        'n' : 'n',
        'o' : 'o',
        'p' : 'p',
        'q' : 'q',
        'r' : 'r',
        's' : 's',
        't' : 't',
        'u' : 'u',
        'v' : 'v',
        'w' : 'w',
        'x' : 'x',
        'y' : 'y',
        'z' : 'z',
        '1' : '1',
        '2' : '2',
        '3' : '3',
        '4' : '4',
        '5' : '5',
        '6' : '6',
        '7' : '7',
        '8' : '8',
        '9' : '9',
        '0' : '0'}

save_dir_videos = 'datasets/source_videos'
save_dir_screenshots = 'datasets/source_images/screenshots'
videos_json_path = os.path.join(save_dir_videos, 'videos.json')
screenshots_json_path = os.path.join(save_dir_screenshots, 'screenshots.json')

if not os.path.exists(save_dir_videos):
    os.makedirs(save_dir_videos)

if not os.path.exists(save_dir_screenshots):
    os.makedirs(save_dir_screenshots)

if not os.path.exists(videos_json_path):
    with open(videos_json_path, 'w') as f:
        json.dump([], f)

if not os.path.exists(screenshots_json_path):
    with open(screenshots_json_path, 'w') as f:
        json.dump([], f)

record = cv2.VideoCapture(0)
frame_width = int(record.get(3))
frame_height = int(record.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # To save in .avi
output_file = None
out = None

is_recording = False
is_croping = False

instructions = """Instructions:
Space: Record
A: Screenshot
Esc: Quit"""

# Function used to create the insctructions that we'll dsplay on screen
def create_instruction_image():
    instruction_image = np.zeros((frame_height, 300, 3), dtype=np.uint8)

    y0, dy = 50, 30
    for i, line in enumerate(instructions.split('\n')):
        y = y0 + i * dy
        cv2.putText(instruction_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return instruction_image

# Function used to give a name to screenshots or videos
def file_name_popup(file_type):
    root = tk.Tk()
    root.withdraw()
    file_name = simpledialog.askstring("File name", f"Enter file name for {file_type}:")
    return file_name

# Function used to create the trace of each screenshot or video
def update_json(json_path, file_info):
    with open(json_path, 'r') as f:
        data = json.load(f)
    data.append(file_info)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

# Main loop
while True:
    if not is_croping:
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

        key = cv2.waitKey(1)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Ask for the label of the video, save it and save a trace in a json
        if key == SPACE:
            if not is_recording:
                video_name = file_name_popup("the video")
                if video_name:
                    file_name = video_name + "_" + current_time + ".avi"
                    output_file = os.path.join(save_dir_videos, file_name)
                    out = cv2.VideoWriter(output_file, fourcc, FPS, (frame_width, frame_height))
                    is_recording = True
                    print("Starting to record...")
            else:
                is_recording = False
                out.release()
                print(f"Recording saved in {output_file}.")
                update_json(videos_json_path, {"filename": output_file, "label": video_name})
        
        #elif key == B_KEY:
        #    is_croping = True
        
        # Quit
        elif key == ESC:
            if not is_recording:
                break
            else:
                print("Stop recording before quitting.")

        # Take a screenshot, ask for the label and save a trace in a json
        for keys in keys_index.keys():
            if (key == ord(keys)):
                print(keys_index[keys])
                image_name = keys_index[keys]
                file_name = image_name + "_" + current_time + ".png"
                output_file = os.path.join(save_dir_screenshots, file_name)
                cv2.imwrite(output_file, frame)
                print(f"Screenshot saved in {output_file}.")
                update_json(screenshots_json_path, {"filename": file_name, "label": image_name})

    # First attempt at switching back and forth between video record and video crop
    else:
        root = tk.Tk()
        app = VideoCropper(root)
        root.mainloop()
        is_croping = False

record.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
