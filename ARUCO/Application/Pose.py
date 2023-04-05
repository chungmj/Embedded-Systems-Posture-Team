"""
BlazePose program.

Author: Gym Sense
Version: 4/2/2023

TODO: combine everything such that the program flows such as before but with the display.
      Continuously append data to file for elbow angle, tilt, etc.
      maybe add in a column of what the system thinks of the form. (elbow is in range yada yada yada)
"""


# TODO: Remove unused imports
import csv
import cv2 as cv
import numpy as np
import sys
import mediapipe as mp
import math
import time
import pytz
from datetime import datetime

from multiprocessing import Process, Queue
import os

# Calculate angle between elbow and body using hip, shoulder, and elbow points
def calculate_angle(a, b, c):
    a = np.array(a)  # hip
    b = np.array(b)  # shoulder
    c = np.array(c)  # elbow

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_blazepose(source, pose_messages, image_messages):
    recording = False
    print(f"Opening BlazePose camera {source}")
    cam2 = cv.VideoCapture(source)
    print("blazepose", cam2.getBackendName(), file=sys.stderr)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    # width = int(cam2.get(cv.CAP_PROP_FRAME_WIDTH))
    # height = int(cam2.get(cv.CAP_PROP_FRAME_HEIGHT))
    # writer = cv.VideoWriter('basicvideo3.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    while True:
        # Grab an image from the camera and send it off before anything else
        ret1, frame1 = cam2.read()
        if ret1:
            image_messages.put((time.time(), "blazepose", frame1), block=False)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

            # writer.write(frame1)

            # Recolor image to RGB
            image = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            # Extract landmarks
            if not (results and results.pose_landmarks and results.pose_landmarks.landmark):
                continue
            landmarks = results.pose_landmarks.landmark

            # Get left body coordinates
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            # Get right body coordinates
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

            # Calculate angle
            angle_l = calculate_angle(hip_l, shoulder_l, elbow_l)
            angle_r = calculate_angle(hip_r, shoulder_r, elbow_r)

            # Visualize angle
            cv.putText(image, str(round(angle_l)),
                       tuple(np.multiply(shoulder_l, [1280, 720]).astype(int)),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA
                       )

            cv.putText(image, str(round(angle_r)),
                       tuple(np.multiply(shoulder_r, [1280, 720]).astype(int)),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA
                       )

            # Shoulder Distance
            x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x - \
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
            y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - \
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            x_r = x * 40.64 / 0.1
            y_r = y * 15.24 / 0.1
            shoulder_distance = math.sqrt(x_r ** 2 + y_r ** 2)

            # print('shoulder distance is: ' + str(shoulder_distance))

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            unix_time = time.time()
            est_tz = pytz.timezone('US/Eastern')
            time_est = datetime.fromtimestamp(unix_time, est_tz)

            # TODO: Send image with the above, copy aruco
            msg_p = (time_est, angle_l, angle_r, shoulder_distance)
            pose_messages.put(msg_p, block=False, timeout=None)

    # cv.VideoCapture(2).release()
    # writer.release()
    # cv.destroyAllWindows()


# Main function
def main():
    # Create queues for message passing
    pose_messages = Queue()
    image_messages = Queue()

    # Create the camera object
    # cam2 = cv.VideoCapture(0)

    # Start the calculate_blazepose thread
    blazepose_thread = Process(target=calculate_blazepose, args=(0, pose_messages, image_messages))
    blazepose_thread.start()

    try:
        filename = 'participant_A.csv'
        with open(filename, 'w', newline='') as file:
            # Create a CSV writer object
            csv_writer = csv.writer(file)

            # Write the header row
            csv_writer.writerow(['Time', 'Left elbow angle', 'Right elbow angle'])

        while True:
            # Receive pose messages
            if not pose_messages.empty():
                msg_p = pose_messages.get()
                timestamp, angle_l, angle_r, shoulder_distance, image = msg_p
                with open(filename, 'a', newline='') as file:
                    # Create a CSV writer object
                    csv_writer = csv.writer(file)

                    # Write some data rows
                    csv_writer.writerow([timestamp, angle_l, angle_r])
                print("Timestamp: ", timestamp, " Angle L: ", angle_l, " Angle R: ", angle_r, " Shoulder Distance: ", shoulder_distance)
                cv.imshow("Pose Image", image)
                cv.waitKey(1)
            if not image_messages.empty():
                msg_i = image_messages.get()
                timestamp, camera_id, image = msg_i

                # Display the image
                cv.imshow("BlazePose Image", image)
                cv.waitKey(1)

    except KeyboardInterrupt:
        # Stop the blazepose_thread and display_images_thread gracefully
        cv.destroyAllWindows()
        blazepose_thread.join()


# if __name__ == "__main__":
    # main()

