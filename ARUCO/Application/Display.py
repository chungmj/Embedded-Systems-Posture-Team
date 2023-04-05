"""
Display program.
Version: 4/3/23
Author: Gym Sense
"""
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk


class RealTimeApplication(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Real Time Application")

        # Create frame for images
        self.img_frame = ttk.Frame(self)
        self.img_frame.grid(row=0, column=0, columnspan=2)

        self.aruco_label = ttk.Label(self.img_frame)
        self.aruco_label.grid(row=0, column=0, padx=10)

        self.pose_label = ttk.Label(self.img_frame)
        self.pose_label.grid(row=0, column=1, padx=10)

        red_img = np.full( (300, 300, 3), (0, 0, 255), dtype=np.uint8)
        green_img = np.full( (300, 300, 3), (0, 255, 0), dtype=np.uint8)
        self.update_aruco_frames(red_img, None)
        self.update_pose_frames(green_img, None)


        # Checkboxes for toggling options
        self.show_pose = tk.BooleanVar()
        self.show_aruco = tk.BooleanVar()


        self.aruco_check = ttk.Checkbutton(self, text="Show Aruco Tags", variable=self.show_aruco)
        self.aruco_check.grid(row=1, column=0, padx=10)

        self.pose_check = ttk.Checkbutton(self, text="Show Pose", variable=self.show_pose)
        self.pose_check.grid(row=1, column=1, padx=10)

        # Variables table
        self.variables_frame = ttk.Frame(self)
        self.variables_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

        self.variables = {
            "Racked" : tk.BooleanVar(value=True),
            "Arched" : tk.BooleanVar(value=False),
            "Grip Width" : tk.DoubleVar(value=0.0),
            "Shoulder Width" : tk.DoubleVar(value=0.0),
            "Left Elbow Angle" : tk.DoubleVar(value=0.0),
            "Right Elbow Angle" : tk.DoubleVar(value=0.0),
            "Bar Tilt" : tk.DoubleVar(value=0.0),
        }

    

        for i, (name, value) in enumerate(self.variables.items()):
            label_name = ttk.Label(self.variables_frame, text=name)
            label_name.grid(row=i, column=0)

            label_value = ttk.Label(self.variables_frame, textvariable=value)
            label_value.grid(row=i, column=1)

    def update_var(self, key, value):
        try:
            self.variables[key].set(value)
        except KeyError:
            print(f"Bad key: {key}")


    def get_var(self, key):
        try:
            return self.variables[key].get()
        except KeyError:
            print(f"Bad key: {key}")

    def update_aruco_frames(self, aruco_raw, aruco_anno):
        """
        Updates the images shown.
        @param aruco_raw the raw aruco image
        @param aruco_raw the aruco image with drawings showing seen tags
        """
        if aruco_anno is not None and self.show_aruco.get():
            self.aruco_label.image = self.resize_image(aruco_anno, 300, 300)
            self.aruco_label.configure(image=self.aruco_label.image)
        elif aruco_raw is not None:
            self.aruco_label.image = self.resize_image(aruco_raw, 300, 300)
            self.aruco_label.configure(image=self.aruco_label.image)


    def update_pose_frames(self, pose_raw, pose_anno):
        if pose_anno is not None and self.show_pose.get():
            self.pose_label.image = self.resize_image(pose_anno, 300, 300)
            self.pose_label.configure(image=self.pose_label.image)
        elif pose_raw is not None:
            self.pose_label.image = self.resize_image(pose_raw, 300, 300)
            self.pose_label.configure(image=self.pose_label.image)

    def resize_image(self, img, width, height):
        img_resized = cv2.resize(img, (width, height))
        return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))

    def on_closing(self):
        self.destroy()


if __name__ == "__main__":
    app = RealTimeApplication()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
