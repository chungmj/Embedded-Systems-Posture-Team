import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

class RealTimeApplication(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Real Time Application")

        # Create frame for images
        self.img_frame = ttk.Frame(self)
        self.img_frame.grid(row=0, column=0, columnspan=2)

        self.cap1 = cv2.VideoCapture(0)
        self.cap2 = cv2.VideoCapture(1)

        self.label1 = ttk.Label(self.img_frame)
        self.label1.grid(row=0, column=0, padx=10)

        self.label2 = ttk.Label(self.img_frame)
        self.label2.grid(row=0, column=1, padx=10)

        self.update_frames()

        # Checkboxes for toggling options
        self.show_pose = tk.BooleanVar()
        self.show_aruco = tk.BooleanVar()

        self.pose_check = ttk.Checkbutton(self, text="Show Pose", variable=self.show_pose)
        self.pose_check.grid(row=1, column=0, padx=10)

        self.aruco_check = ttk.Checkbutton(self, text="Show Aruco Tags", variable=self.show_aruco)
        self.aruco_check.grid(row=1, column=1, padx=10)

        # Variables table
        self.variables_frame = ttk.Frame(self)
        self.variables_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

        self.variables = [
            ("Racked", True),
            ("Grip Width", 25.0),
            ("Left Elbow Angle", 90.0),
            ("Right Elbow Angle", 90.0),
            ("Bar Tilt", 0.0),
        ]

        for i, (name, value) in enumerate(self.variables):
            label_name = ttk.Label(self.variables_frame, text=name)
            label_name.grid(row=i, column=0)

            label_value = ttk.Label(self.variables_frame, text=value)
            label_value.grid(row=i, column=1)

    def update_frames(self):
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        if ret1 and ret2:
            self.label1.image = self.resize_image(frame1, 300, 300)
            self.label1.configure(image=self.label1.image)

            self.label2.image = self.resize_image(frame2, 300, 300)
            self.label2.configure(image=self.label2.image)

        self.after(20, self.update_frames)

    def resize_image(self, img, width, height):
        img_resized = cv2.resize(img, (width, height))
        return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))

    def on_closing(self):
        self.cap1.release()
        self.cap2.release()
        self.destroy()

if __name__ == "__main__":
    app = RealTimeApplication()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
