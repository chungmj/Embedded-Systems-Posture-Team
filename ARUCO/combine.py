import cv2 as cv
from cv2 import aruco
import numpy as np
import sys
import mediapipe as mp
import serial.tools.list_ports
import serial
import math

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

portList = []

for onePort in ports:
    portList.append(str(onePort))
    print(str(onePort))

portVar = '/dev/cu.usbserial-140'

serialInst.baudrate = 9600
serialInst.port = portVar
serialInst.open()




#bar path
calib_data_path = "../calib_data/MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

# MARKER_SIZE = 9.97331  # centimeters
MARKER_SIZE = 5  # centimeters

marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)

param_markers = aruco.DetectorParameters_create()

cap = cv.VideoCapture(1)
cap2 = cv.VideoCapture(0)

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


def get_Arduino_data():
    return None

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        #multithreading / multiprocessing. mutex

        #get_Arduino_data()
        #if serialInst.in_waiting:

            # send '1' to Arduino


            # parse the response
        packet = serialInst.readline()
        print(packet.decode('ISO-8859-1').rstrip('\n'))
            #serial.write('/')
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

            A = np.where(marker_IDs == 25)
            B = np.where(marker_IDs == 69)
            C = np.where(marker_IDs == 14)
            # handle aruco tag not found
            locB = -tVec[B]
            locC = -tVec[C]
            dist = np.linalg.norm(locB - locC)
            #print('Grip width is: ' + str(dist) + ' cm')

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

                # Calculating the distance
                distance = np.sqrt(
                    tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )
                #print(tVec[i])
                # Draw the pose of the marker
                point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)

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
        #print('shoulder dist is: ' + str(shoulder_distance))
        # if distance > shoulder_distance*2:
        #     print('GRIP WIDTH IS TOO WIDE')
        # elif distance < shoulder_distance*1.5:
        #     print('GRIP WIDTH IS TOO NARROW')
        print('shoulder distance is: ' + str(shoulder_distance) + ' and grip width is: ' + str(distance))



        # except:
        #     pass

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
