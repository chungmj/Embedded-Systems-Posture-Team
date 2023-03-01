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
import matplotlib.pyplot as plt

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

portVar = '/dev/cu.usbserial-1130'

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

UNRACKED = 5.0

# Aruco Marker specifications
MARKER_SIZE = 9.97331  # centimeters
MARKER_SIZE_B = 5  # centimeters
marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
param_markers = aruco.DetectorParameters_create()

#
cap = cv.VideoCapture(1)
cap2 = cv.VideoCapture(2)
# cap3 = cv.VideoCapture(0)

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
            # else:
            #     print('detected no markers', file=sys.stderr)

            # if ret:
            #     cv.imshow("frame", frame)


def calculate_blazepose(record_request2, pose_messages):
    recording = False
    while not shutdown:
        try:
            request = record_request2.get(block=False)
            print(request)
            recording = request
        except queue.Empty:
            # print('no request')
            if not recording:
                continue
        if recording:
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
                if not landmarks:
                    continue

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
                msg_p = (time.time(), angle_l, angle_r, shoulder_distance)
                pose_messages.put(msg_p, block=False, timeout=None)


def calculate_barPath(record_request3, bar_messages):
    recording = False
    while not shutdown:
        try:
            request = record_request3.get(block=False)
            print(request)
            recording = request
        except queue.Empty:
            if not recording:
                continue
        if recording:
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
                    msg_bar = (time.time(), locA)  # Message is a tuple of (current time, grip width)
                    bar_messages.put(msg_bar, block=False, timeout=None)


def calculate_arduino(arduino_messages):
    last_contact = UNRACKED

    while not shutdown:
        if ser.in_waiting:
            # *THIS "ATTEMPT" ALWAYS SUCCEEDS BECAUSE IT WILL BLOCK UNTIL IT WORKS*
            try:
                packet = ser.readline()
                txt = packet.decode('ISO-8859-1').rstrip('\n').rstrip('\r')
                arrayTxt = txt.split(",")
                roll_rad = math.atan2(float(arrayTxt[1]), float(arrayTxt[2]))
                roll_angle = math.degrees(roll_rad)
                roll_angle_arr.append(roll_angle)

                contact = float(arrayTxt[3])
                contact = int(contact)
                contact == UNRACKED
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
pose_record_request = queue.Queue()
bar_record_request = queue.Queue()

grip_messages = queue.Queue()
pose_messages = queue.Queue()
bar_messages = queue.Queue()
arduino_messages = queue.Queue()
rack_state = queue.Queue()
accel_messages = queue.Queue()

t0 = threading.Thread(target=calculate_arduino, args=(arduino_messages,))
t0.start()

t1 = threading.Thread(target=calculate_grip_width, args=(grip_record_request, grip_messages))
t1.start()

t2 = threading.Thread(target=calculate_blazepose, args=(pose_record_request, pose_messages))
t2.start()

# t3 = threading.Thread(target=calculate_barPath, args=(bar_record_request, bar_messages))
# t3.start()

grip_dist_sum = 0
grip_dist = 0
shoulder_dist = 0
grip_msg_received = 0
shoulder_dist_sum = 0
shoulder_msg_received = 0
left_angle_sum = 0
left_angle_received = []
right_angle_sum = 0
right_angle_received = []
left_angle_data = []
right_angle_data = []
roll_angle_arr = []

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
        print('racked = ', str(racked))
        if racked:
            print('sending request')
            grip_record_request.put(racked)
            # bar_record_request.put(racked)
            pose_record_request.put(racked)

            # if grip_dist > 0 and shoulder_dist > 0:
            #     if grip_dist > 1.5 * shoulder_dist:
            #         dist_error = grip_dist - (1.5 * shoulder_dist)
            #         print('Grip width is', dist_error, 'cm too wide!')
            #     elif grip_dist < shoulder_dist:
            #         dist_error = shoulder_dist - grip_dist
            #         print('Grip width is', dist_error, 'cm too narrow!')
            #     else:
            #         print('Grip width is within range')
        else:
            if grip_msg_received > 0 and shoulder_msg_received > 0:
                grip_dist_avg = grip_dist_sum / grip_msg_received
                shoulder_dist_avg = shoulder_dist_sum / shoulder_msg_received
                if grip_dist_avg > 1.5 * shoulder_dist_avg:
                    print(f'Grip width is too wide. Your grip width is: {grip_dist_avg:.2f}')
                elif grip_dist_avg < shoulder_dist_avg:
                    print(f'Grip width is too narrow. Your grip width is: {grip_dist_avg:.2f}')
                else:
                    print(f'Your grip width is within the ideal range. Grip width: {grip_dist_avg:.2f}')


                left_angle_sum = sum(left_angle_data)
                left_avg = left_angle_sum / len(left_angle_data)
                right_angle_sum = sum(right_angle_data)
                right_avg = right_angle_sum / len(right_angle_data)
                left_std = np.std(left_angle_data)
                right_std = np.std(right_angle_data)
                if left_avg > 75:
                    print(f'Too wide. Your average left elbow angle is: {left_avg:.2f}')
                elif left_avg < 45:
                    print(f'Too narrow. Your average left elbow angle is: {left_avg:.2f}')
                else:
                    print(f'Your elbow angle is within the ideal range. Your average left elbow angle is:'
                          f' {left_avg:.2f}')

                if right_avg > 75:
                    print(f'Too wide. Your average right elbow angle is: {right_avg:.2f}')
                elif right_avg < 45:
                    print(f'Too narrow. Your average right elbow angle is: {right_avg:.2f}')
                else:
                    print(f'Your elbow angle is within the ideal range. Your average right elbow angle is:'
                          f' {right_avg:.2f}')

                roll_avg = sum(roll_angle_arr) / len(roll_angle_arr)
                print(f'Average tilt angle is: {roll_avg:.2f}')
                print(f'Left standard deviation: {left_std:.2f}')
                print(f'Right standard deviation: {right_std:.2f}')

            grip_dist_sum = 0
            grip_msg_received = 0
            shoulder_dist_sum = 0
            shoulder_msg_received = 0
            left_angle_sum = 0
            right_angle_sum = 0
            left_avg = 0
            right_avg = 0

    except queue.Empty:
        pass

    try:
        grip_data = grip_messages.get(block=False)
        # cv.imshow("frame", grip_data[2])
        # print('grip width: ' + str(grip_data[1]))
        grip_dist_sum += grip_data[1]
        grip_msg_received += 1
        grip_dist = grip_data[1]

    except queue.Empty:
        pass

    try:
        pose_data = pose_messages.get(block=False)
        # print('left elbow angle ' + str(pose_data[1]))
        left_angle_data.append(pose_data[1])
        # print('right elbow angle ' + str(pose_data[2]))
        right_angle_data.append(pose_data[2])
        # print('shoulder distance: ' + str(pose_data[3]))
        shoulder_dist_sum += pose_data[3]
        shoulder_msg_received += 1
        shoulder_dist = pose_data[3]

    except queue.Empty:
        pass

    # try:
    #     bar_data = bar_messages.get(block=False)
    #     print(bar_data[1])
    # except queue.Empty:
    #     pass

t0.join()
t1.join()
t2.join()
# t3.join()

"""
Write data to file
Once RERACKED, process data
once calculated, present results

Tkinter >>> "Start set" > silently collect data > rerack > present results

Results:
- reps
- successes / failures / tips?
- path of bar?

Wants: calculate data, once reracked >>> print success / failures
bar path
"""

# pressure sensor > voltage divider
# ui
# bar path >> similarity of curves. dynamic time warping. spline matching. sometimes turn each segment a letter and turn curves into sequences
# spline match if time isn't important

# can do balance / slope of bar using aruco tags

# loop for graph. thread for elbows collects elbow data as well as a counter or time
