import TimeAlign
import time
import cv2
import pyautogui
import numpy as np


def capture_image(cap):
    ret, frame = cap.read()
    if ret:
        return frame
    else:
        print("Failed to capture image")
        return None

def capture_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot_array = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot_array

def my_callback(miu_time, aligned_data):
    print(f"Data ready at MIU time {miu_time}")

    # Extract the images from the aligned_data
    images = [data for _, data in aligned_data]

    # Check if there are any images to concatenate
    if len(images) > 1:
        # Resize images to have the same height
        max_height = max(img.shape[0] for img in images)
        resized_images = [cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height)) for img in images]

        # Concatenate images horizontally
        stitched_image = cv2.hconcat(resized_images)

        # Display the stitched image
        cv2.imshow('Stitched Images', stitched_image)
        cv2.waitKey(10)  # Display the image for a short period (1 ms)
    elif len(images) == 1:
        # Display the single image if there's only one
        cv2.imshow('Stitched Images', images[0])
        cv2.waitKey(10)


sample_rates = [30, 30]  # Sample rates for your devices (in Hz)
ready_count = 2  # Number of samples to be aligned before the data is considered ready
time_aligner = TimeAlign.TimeAligner(sample_rates, ready_count, my_callback)
time_aligner.start()

# Simulate data coming in real-time
try:
    cap = cv2.VideoCapture(0)  # Open the camera once outside the loop

    while True:
        time.sleep(1 / sample_rates[0])
        timestamp = int(time.time() * 1e6)

        # Capture image from the camera and add to TimeAligner
        camera_image = capture_image(cap)
        if camera_image is not None:
            camera_image = cv2.resize(camera_image, (720, 720), interpolation = cv2.INTER_AREA)
            time_aligner.add_data(0, timestamp, camera_image)


        # Capture a screenshot and add to TimeAligner
        screenshot = capture_screenshot()
        if screenshot is not None:
            screenshot = cv2.resize(screenshot, (720, 720), interpolation = cv2.INTER_AREA)
            time_aligner.add_data(1, timestamp, screenshot)



except KeyboardInterrupt:
    print("Stopping time aligner...")
    time_aligner.stop()