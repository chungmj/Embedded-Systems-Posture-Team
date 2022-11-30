import cv2 as cv
from cv2 import aruco
import numpy as np
import sys
import mediapipe as mp
import serial.tools.list_ports
import serial
import math
import threading
import queue
import time
import random
import signal

shutdown = False


def signal_handler():
    print('You pressed Ctrl+C!')
    shutdown = True


signal.signal(signal.SIGINT, signal_handler)

# Arduino Set up
ports = serial.tools.list_ports.comports()
ser = serial.Serial()

portList = []

for onePort in ports:
    portList.append(str(onePort))
    print(str(onePort))

portVar = '/dev/cu.usbserial-1140'

ser.baudrate = 9600
ser.port = portVar
ser.open()

# bar path
calib_data_path = "../calib_data/MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

MARKER_SIZE = 9.97331  # centimeters
MARKER_SIZE_B = 5  # centimeters

UNRACKED = 0
marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)

param_markers = aruco.DetectorParameters_create()

cap = cv.VideoCapture(1)
cap2 = cv.VideoCapture(0)
# cap3 = cv.VideoCapture(2)

try:
    fp = open('out.csv', 'w')
except IOError:
    print('error')
    sys.exit(-1)

# mediapipe / blazepose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


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


def calculate_grip_width(record_request, grip_messages):
    recording = False
    while not shutdown:
        try:
            request = record_request.get(block=False)
            print(request)
            recording = request
        except queue.Empty:
            # print('no request')
            if not recording:
                continue
        if recording:
            ret, frame = cap.read()
            # print(time.time(), file=sys.stderr)
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
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
                # print('Grip width is: ' + str(dist) + ' cm')

                fp.flush()

                corners = marker_corners[0].reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                top_left = corners[1].ravel()
                bottom_right = corners[2].ravel()
                bottom_left = corners[3].ravel()

                for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
                    cv.polylines(
                        frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
                    )
                    corners = corners.reshape(4, 2)
                    corners = corners.astype(int)
                    top_right = corners[0].ravel()
                    top_left = corners[1].ravel()
                    bottom_right = corners[2].ravel()
                    bottom_left = corners[3].ravel()

                    # Draw the pose of the marker
                    point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
                    # print('grip width is: ' + str(dist), file=sys.stderr)
                    msg = (time.time(), dist, frame)  # Message is a tuple of (current time, grip width)
                    grip_messages.put(msg, block=False, timeout=None)
            else:
                print('detected no markers', file=sys.stderr)

            # if ret:
            #     cv.imshow("frame", frame)


def calculate_blazepose(pose_record_request):
    while True:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            ret1, frame1 = cap2.read()
            # Recolor image to RGB
            image = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            # Extract landmarks
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

            print('shoulder distance is: ' + str(shoulder_distance))

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            # if ret1:
            #     cv.imshow('Mediapipe Feed', image)


def calculate_barPath():
    ret, frame = cap3.read()

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE_B, cam_mat, dist_coef
        )

        total_markers = range(0, marker_IDs.size)

        A = np.where(marker_IDs == 25)

        locA = tVec[A]

        fp.flush()

        corners = marker_corners[0].reshape(4, 2)
        corners = corners.astype(int)
        top_right = corners[0].ravel()
        top_left = corners[1].ravel()
        bottom_right = corners[2].ravel()
        bottom_left = corners[3].ravel()

        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # print(tVec[i])
            # Draw the pose of the marker
            point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            print(locA)


def calculate_arduino(arduino_messages):
    last_contact = not UNRACKED

    while not shutdown:
        if ser.in_waiting:
            # *THIS "ATTEMPT" ALWAYS SUCCEEDS BECAUSE IT WILL BLOCK UNTIL IT WORKS*
            try:
                packet = ser.readline()
                # print(packet.decode('ISO-8859-1').rstrip('\n'))
                txt = packet.decode('ISO-8859-1').rstrip('\n')
                arrayTxt = txt.split(",")
                #print(arrayTxt[3])

                contact = int(arrayTxt[3]) == UNRACKED
                if last_contact != UNRACKED and contact == UNRACKED:
                    print('Bar Racked')
                    rack_state.put(False)

                elif last_contact == UNRACKED and contact != UNRACKED:
                    print('Bar Unracked')
                    rack_state.put(True)

                last_contact = contact

            except IndexError:
                print('ignoring error')


grip_record_request = queue.Queue()
# pose_record_request = queue.Queue()
# bar_record_request = queue.Queue()

grip_messages = queue.Queue()
arduino_messages = queue.Queue()
rack_state = queue.Queue()
accel_messages = queue.Queue()

t0 = threading.Thread(target=calculate_arduino, args=(arduino_messages,))
t0.start()

t1 = threading.Thread(target=calculate_grip_width, args=(grip_record_request, grip_messages))
t1.start()

# t2 = threading.Thread(target=calculate_blazepose, args=(pose_record_request,))
# t2.start()

# t3 = threading.Thread(target=calculate_barPath(), args=())
# t3.start()

while True:
    # TODO #1: Make a new thread for the arduino, having the main processing on this thread is
    #          difficult because of the baud rate of the arduino being much lower than the cams

    # Alternatively, you could have the "main" thread be one of the cameras but I feel like this
    # makes the processing sloppy since you'd "send" a message to yourself

    # -----------------------------------------------------------------------------------------

    # TODO #2: Synchronize the reading of messages ( by serializing them )
    # Attempt to read arduino (store to variable)
    # Attempt to read grip message (store to variable)

    # Process arduino message
    # Process grip message

    # Should do this because it's confusing to hook up the error path (failed to read msg)
    # to attempt to read the next message

    # For example, arduino message fails, so in the except block we try to read grip message
    # ^ this leads to a bunch of spaghetti code so I'd put all of the attempts and then process

    # TODO #3: For any sort of user interface, i'd recommend using tkinter which can make
    #  a desktop app appear vs creating some sort of website ui with flask/django (because it'd be harder imo)
    try:
        racked = rack_state.get(block=False)
        print(str(racked) + ' on the main thread')
        grip_record_request.put(racked)

    except queue.Empty:
        # print('no rack state')
        pass

    try:
        message = grip_messages.get(block=False)
        print(message[1])
    except queue.Empty:
        pass

t0.join()
t1.join()
# t2.join()
# t3.join()
