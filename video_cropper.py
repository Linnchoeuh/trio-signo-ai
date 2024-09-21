import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import os

class VideoTrimmer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Trimmer")
        self.video_path = ""
        self.video_clip = None
        self.current_frame = None
        self.playing = False
        
        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()

        # Load and save buttons
        self.load_button = tk.Button(root, text="Load Video", command=self.load_video)
        self.load_button.pack()
        self.trim_button = tk.Button(root, text="Save video", command=self.trim_video, state=tk.DISABLED)
        self.trim_button.pack()

        # Sliders to crop time
        self.start_slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, label="Start", length=500)
        self.start_slider.pack()
        self.end_slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, label="End", length=500)
        self.end_slider.pack()


    def load_video(self):
        self.video_path = filedialog.askopenfilename(initialdir="datasets/source_videos", filetypes=[("Video Files", "*.avi *.mp4")])
        if self.video_path:
            self.video_clip = VideoFileClip(self.video_path)
            self.update_sliders()
            self.trim_button.config(state=tk.NORMAL)
            self.play_video()

    def update_sliders(self):
        duration = self.video_clip.duration
        self.start_slider.config(to=duration)
        self.end_slider.config(to=duration)
        self.end_slider.set(duration)

    def play_video(self):
        self.playing = True
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened() and self.playing:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (600, 400))
                self.current_frame = frame

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = np.array(img)
                img = np.flip(img, axis=2)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                self.show_frame(img)
                cv2.waitKey(33)
            else:
                break
        cap.release()

    def show_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        img = np.flip(img, axis=2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.photo = tk.PhotoImage(master=self.canvas, width=600, height=400)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def trim_video(self):
        start_time = self.start_slider.get()
        end_time = self.end_slider.get()

        if start_time >= end_time:
            messagebox.showerror("Error", "Wrong trim.")
            return

        trimmed_clip = self.video_clip.subclip(start_time, end_time)
        save_path = filedialog.asksaveasfilename(initialdir="datasets/source_videos", defaultextension=".avi", filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4")])
        if save_path.endswith(".avi"):
            trimmed_clip.write_videofile(save_path, codec="mpeg4")
        else:
            trimmed_clip.write_videofile(save_path, codec="libx264")
        messagebox.showinfo("Success", "Video saved successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTrimmer(root)
    root.mainloop()
