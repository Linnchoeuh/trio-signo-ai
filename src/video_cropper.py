import cv2
import tkinter as tk
from PIL import Image, ImageTk
from moviepy.editor import VideoFileClip
from tkinter import filedialog, messagebox

class VideoCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Cropper")
        self.video_path = ""
        self.video_clip = None
        self.current_frame = None
        self.playing = False

        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()

        self.load_button = tk.Button(root, text="Load video", command=self.load_video)
        self.load_button.pack()
        
        self.trim_button = tk.Button(root, text="Trim video", command=self.trim_video, state=tk.DISABLED)
        self.trim_button.pack()
        
        self.split_button = tk.Button(root, text="Split video", command=self.split_video, state=tk.DISABLED)
        self.split_button.pack()

        self.start_slider = tk.Scale(root, from_=0, to=1, orient=tk.HORIZONTAL, label="Start", length=500, resolution=0.01, showvalue=0, command=self.update_start_label)
        self.start_slider.pack()
        self.end_slider = tk.Scale(root, from_=0, to=1, orient=tk.HORIZONTAL, label="End", length=500, resolution=0.01, showvalue=0, command=self.update_end_label)
        self.end_slider.pack()

    def load_video(self):
        self.video_path = filedialog.askopenfilename(initialdir="datasets", filetypes=[("Video Files", "*.avi *.mp4")])
        if self.video_path:
            self.video_clip = VideoFileClip(self.video_path)
            self.update_sliders()
            self.trim_button.config(state=tk.NORMAL)
            self.split_button.config(state=tk.NORMAL)
            self.play_video()

    def update_sliders(self):
        duration = self.video_clip.duration
        self.start_slider.config(to=duration)
        self.end_slider.config(to=duration)
        self.end_slider.set(duration)

    def update_start_label(self, value):
        self.start_slider.config(label=f"Start : {float(value):.2f} sec")

    def update_end_label(self, value):
        self.end_slider.config(label=f"End : {float(value):.2f} sec")

    def play_video(self):
        self.playing = True
        cap = cv2.VideoCapture(self.video_path)

        def update_frame():
            if cap.isOpened() and self.playing:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (600, 400))
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                    self.canvas.image = imgtk
                    self.root.after(33, update_frame)
                else:
                    cap.release()

        update_frame()

    def trim_video(self):
        start_time = float(self.start_slider.get())
        end_time = float(self.end_slider.get())

        if start_time >= end_time:
            messagebox.showerror("Error", "Invalid crop.")
            return

        trimmed_clip = self.video_clip.subclip(start_time, end_time)
        save_path = filedialog.asksaveasfilename(initialdir="datasets/source_videos", defaultextension=".avi", filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4")])
        if save_path.endswith(".avi"):
            trimmed_clip.write_videofile(save_path, codec="mpeg4")
        else:
            trimmed_clip.write_videofile(save_path, codec="libx264")
        messagebox.showinfo("Success", "Video saved.")

    def split_video(self):
        split_time = float(self.start_slider.get())

        if split_time <= 0 or split_time >= self.video_clip.duration:
            messagebox.showerror("Error", "Invalid split time.")
            return

        first_part = self.video_clip.subclip(0, split_time)
        second_part = self.video_clip.subclip(split_time, self.video_clip.duration)

        first_part_path = filedialog.asksaveasfilename(initialdir="datasets/source_videos", defaultextension=".avi", filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4")])
        if first_part_path:
            if first_part_path.endswith(".avi"):
                first_part.write_videofile(first_part_path, codec="mpeg4")
            else:
                first_part.write_videofile(first_part_path, codec="libx264")

        second_part_path = filedialog.asksaveasfilename(initialdir="datasets/source_videos", defaultextension=".avi", filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4")])
        if second_part_path:
            if second_part_path.endswith(".avi"):
                second_part.write_videofile(second_part_path, codec="mpeg4")
            else:
                second_part.write_videofile(second_part_path, codec="libx264")

        messagebox.showinfo("Success", "Video split and saved.")

# Create the main window
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCropper(root)
    root.mainloop()
