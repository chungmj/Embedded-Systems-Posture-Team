import tkinter as tk
from tkinter import ttk

# Create the main window and use the ttk.Style class to set the theme
root = tk.Tk()
style = ttk.Style(root)
style.theme_use("aqua")
root.geometry('1440x1080')

# Create a label and a button using the ttk widgets
label = ttk.Label(root, text="Hello, world!")

# Use the grid layout manager to arrange the widgets in the UI
label.grid(row=0, column=0)

# Start the main event loop
root.mainloop()
