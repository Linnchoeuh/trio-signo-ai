import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class VideoTrimmer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Trimmer")
        self.video_path = ""
        self.video_clip = None
        self.current_frame = None
        self.playing = False

        # Interface graphique
        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()

        # Boutons
        self.load_button = tk.Button(root, text="Charger Vidéo", command=self.load_video)
        self.load_button.pack()
        self.trim_button = tk.Button(root, text="Sauvegarder la Vidéo", command=self.trim_video, state=tk.DISABLED)
        self.trim_button.pack()

        # Curseurs de découpage avec affichage en secondes
        self.start_slider = tk.Scale(root, from_=0, to=1, orient=tk.HORIZONTAL, label="Début : 0.00 sec", length=500, resolution=0.01, showvalue=0, command=self.update_start_label)
        self.start_slider.pack()
        self.end_slider = tk.Scale(root, from_=0, to=1, orient=tk.HORIZONTAL, label="Fin : 0.00 sec", length=500, resolution=0.01, showvalue=0, command=self.update_end_label)
        self.end_slider.pack()

    def load_video(self):
        self.video_path = filedialog.askopenfilename(initialdir="datasets/source_videos", filetypes=[("Video Files", "*.avi *.mp4")])
        if self.video_path:
            self.video_clip = VideoFileClip(self.video_path)
            self.update_sliders()
            self.trim_button.config(state=tk.NORMAL)
            self.play_video()

    def update_sliders(self):
        # Mettre à jour les curseurs en fonction de la durée de la vidéo, sans multiplier par 100
        duration = self.video_clip.duration
        self.start_slider.config(to=duration)
        self.end_slider.config(to=duration)
        self.end_slider.set(duration)

    def update_start_label(self, value):
        # Mettre à jour dynamiquement l'étiquette du curseur de début
        self.start_slider.config(label=f"Début : {float(value):.2f} sec")

    def update_end_label(self, value):
        # Mettre à jour dynamiquement l'étiquette du curseur de fin
        self.end_slider.config(label=f"Fin : {float(value):.2f} sec")

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
                    self.root.after(33, update_frame)  # Appel périodique toutes les 33 ms pour lecture à 30 fps
                else:
                    cap.release()

        update_frame()

    def trim_video(self):
        # Récupérer les valeurs des curseurs en secondes (pas besoin de conversion maintenant)
        start_time = float(self.start_slider.get())
        end_time = float(self.end_slider.get())

        if start_time >= end_time:
            messagebox.showerror("Erreur", "Le découpage est incorrect.")
            return

        trimmed_clip = self.video_clip.subclip(start_time, end_time)
        save_path = filedialog.asksaveasfilename(initialdir="datasets/source_videos", defaultextension=".avi", filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4")])
        if save_path.endswith(".avi"):
            trimmed_clip.write_videofile(save_path, codec="mpeg4")
        else:
            trimmed_clip.write_videofile(save_path, codec="libx264")
        messagebox.showinfo("Succès", "La vidéo a été sauvegardée avec succès.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTrimmer(root)
    root.mainloop()
