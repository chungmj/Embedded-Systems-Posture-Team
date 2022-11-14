import cv2 as cv
from cv2 import aruco
import numpy as np
import sys
import mediapipe as mp

calib_data_path = "../calib_data/MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

#MARKER_SIZE = 9.97331  # centimeters
MARKER_SIZE = 5 #centimeters

marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)

param_markers = aruco.DetectorParameters_create()

cap = cv.VideoCapture(0)


try:
    fp = open('out.csv', 'w')
except IOError:
    print('error')
    sys.exit(-1)

while True:
    ret, frame = cap.read()
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

        # solvePnP is recommended from stack overflow to improve detection
        # https://stackoverflow.com/questions/73276708/how-to-improve-aruco-marker-pose-estimation
        #rVec, tVec, _ = aruco.SolvePnP(marker_corners, MARKER_SIZE, cam_mat, dist_coef)
        total_markers = range(0, marker_IDs.size)

        A = np.where(marker_IDs == 25)

        locA = tVec[A]
        # print(str(locA[0, 0]) + ',' + str(locA[0, 1]) + ',' + str(locA[0, 2]), file=fp)
        # print(str(locA[0, 0]) + ',' + str(locA[0, 1]) + ',' + str(locA[0, 2]))

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

            #print(tVec[i])
            # Draw the pose of the marker
            point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)

    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
fp.close()


