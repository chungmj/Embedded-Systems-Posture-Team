"""
Name: CAPSTONE TEAM
Date: 4/2/2023
"""

import numpy as np
import cv2 as cv
from cv2 import aruco
import time
import os

from multiprocessing import Queue, Process

# Calibration for Aruco cameras
calib_data_path = "../calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]



# Aruco Marker specifications
MARKER_SIZE = 9.97331  # centimeters
MARKER_SIZE_B = 5  # centimeters
marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
param_markers = aruco.DetectorParameters_create()


def calculate_grip_width(source, grip_messages, image_messages):
    """
    This function sends out messages for the participant's grip width and
    raw camera data.

    Grip messages are in this format:
        (
          Current time,
          Grip width,
          Bar Tilt
        )

    Images messages are in this format:
        (
          Current time,
          Raw data,
          Aruco tag annotation image
        )

    @param cap           cv2 VideoCapture object
    @param grip_messages  the message queue to send grip messages over
    @param image_messages the message queue to send raw images messages over

    """
    print(f"Opening aruco camera {source}...")
    cap = cv.VideoCapture(source)
    print(f"Opened aruco camera {source}")

    delay = 1 / 60
    while True:
        # Throttle the camera to save FLOPS
        time.sleep(delay)

        start = time.time()
        ret, raw = cap.read()
        annotated = None

        if ret:
            annotated = raw.copy()

            downscale = 0.55
            img_resized = cv.resize(raw, (int(raw.shape[0] * downscale), int(raw.shape[1] * downscale)))
            gray_frame = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)

            marker_corners, marker_IDs, reject = aruco.detectMarkers(
                gray_frame, marker_dict, parameters=param_markers
            )

            if marker_corners:
                rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                    marker_corners, MARKER_SIZE, cam_mat, dist_coef
                )
                total_markers = range(0, marker_IDs.size)

                B = np.where(marker_IDs == 69)
                C = np.where(marker_IDs == 14)
                # handle aruco tag not found
                locB = -tVec[B]
                locC = -tVec[C]
                dist = np.linalg.norm(locB - locC)
                if locB.shape[0] == 0 or locC.shape[0] == 0:
                    continue
                dist = np.linalg.norm(locB - locC)

                vB = np.squeeze(np.asarray(locB))
                vC = np.squeeze(np.asarray(locC))

                vector = vC - vB
                angle_rad = np.arctan2(vector[1], np.linalg.norm(vector[[0, 2]]))
                angle_deg = np.degrees(angle_rad)
                # print(angle_deg)

                corners = marker_corners[0].reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                top_left = corners[1].ravel()
                bottom_right = corners[2].ravel()
                bottom_left = corners[3].ravel()

                for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
                    cv.polylines(
                        annotated, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
                    )
                    corners = corners.reshape(4, 2)
                    corners = corners.astype(int)
                    top_right = corners[0].ravel()
                    top_left = corners[1].ravel()
                    bottom_right = corners[2].ravel()
                    bottom_left = corners[3].ravel()

                    # Draw the pose of the marker
                    point = cv.drawFrameAxes(annotated, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
                    # print('grip width is: ' + str(dist), file=sys.stderr)
                    msg = (time.time(), dist, angle_deg)  # Message is a tuple of (current time, grip width)
                    grip_messages.put(msg)
        image_messages.put((time.time(), "grip", raw, annotated))


def init_video_writer(width, height, fps, output_filename):
    """
    Initialize a video writer.

    @param width:          video width
    @param height:         video height
    @param fps:            frames per second
    @param output_filename: video output filename
    @return:               video writer object
    """
    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    return cv.VideoWriter(output_filename, fourcc, fps, (width, height))


# if __name__ == "__main__":
# 
#     grip_messages = Queue()
#     image_messages = Queue(maxsize=20)
# 
#     t1 = Process(target=calculate_grip_width, \
#         args=(0, grip_messages, image_messages))
#     t1.start()
# 
#     video_writer = None
#     video_fps = 30
#     video_output_filename = "output_video.avi"
# 
#     try:
#         while True:
#             # Get image messages
#             try:
#                 t, tag, raw, annotated = image_messages.get(block=False)
#                 print("Queue size: ", image_messages.qsize())
#                 if raw is not None:
#                     cv.imshow("Raw", raw)
#                 if annotated is not None:
#                     cv.imshow("Annotate", annotated)
# 
#                     # Initialize the video writer if it's not initialized
#                     if video_writer is None:
#                         height, width, _ = annotated.shape
#                         video_writer = init_video_writer(width, height, video_fps, video_output_filename)
# 
#                     # Write the annotated frame to the video file
#                     video_writer.write(annotated)
# 
#                 if annotated is not None or raw is not None:
#                     cv.waitKey(1)
#             except queue.Empty:
#                 pass
# 
#             # Get grip messages
#             try:
#                 message = grip_messages.get(block=False)
#                 # print("Grip: ", message)
#             except queue.Empty:
#                 pass
# 
#     except KeyboardInterrupt:
#         pass
# 
#     # Release the video writer and save the video file
#     if video_writer is not None:
#         video_writer.release()
# 
#     # Since we made the other thread a daemon, we don't need to join it!
#     # t1.join()
# 
# 
