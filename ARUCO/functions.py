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
                # print(packet.decode('ISO-8859-1').rstrip('\n'))
                txt = packet.decode('ISO-8859-1').rstrip('\n').rstrip('\r')
                arrayTxt = txt.split(",")
                #print(arrayTxt[3])

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