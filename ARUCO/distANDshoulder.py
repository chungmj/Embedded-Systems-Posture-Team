import cv2 as cv
from cv2 import aruco
import numpy as np
import sys
import mediapipe as mp
import serial.tools.list_ports


#bar path
calib_data_path = "../calib_data/MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

MARKER_SIZE = 9.97331  # centimeters

marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)

param_markers = aruco.DetectorParameters_create()

cap = cv.VideoCapture(0)
cap2 = cv.VideoCapture(1)

try:
    fp = open('out.csv', 'w')
except IOError:
    print('error')
    sys.exit(-1)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# makes the camera display 720p
def make_720p():
    cap2.set(3, 1280)
    cap2.set(4, 720)


make_720p()


# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        ret1, frame1 = cap2.read()

        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, reject = aruco.detectMarkers(
            gray_frame, marker_dict, parameters=param_markers
        )
        if marker_corners:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                marker_corners, MARKER_SIZE, cam_mat, dist_coef
            )
            total_markers = range(0, marker_IDs.size)

            A = np.where(marker_IDs == 69)
            B = np.where(marker_IDs == 14)

            locA = -tVec[A]
            locB = -tVec[B]
            dist = np.linalg.norm(locA - locB)
            print(str(dist) + ' cm')

            corners = marker_corners[0].reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            frame = cv.putText(
                frame,
                f"Delta{round(dist, 2)}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (255, 0, 255),
                2,
                cv.LINE_AA,
            )

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

                # Calculating the distance
                distance = np.sqrt(
                    tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )
                print(tVec[i])
                # Draw the pose of the marker
                point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Shoulder Distance
            x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x - \
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
            y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - \
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            x_r = x * 40.64 / 0.1
            y_r = y * 15.24 / 0.1
            shoulder_distance = math.sqrt(x_r ** 2 + y_r ** 2)
            print(shoulder_distance)

        if ret:
            cv.imshow("frame", frame)


        # Recolor image to RGB
        image = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        if ret1:
            cv.imshow('Mediapipe Feed', image)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap2.release()
    cv.destroyAllWindows()
