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
from playsound import playsound
from gtts import gTTS
import select


def heardEnter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            input = sys.stdin.readline()
            return True
    return False


shutdown = False


# Clean Shutdown function
def signal_handler():
    print('You pressed Ctrl+C!')
    shutdown = True


signal.signal(signal.SIGINT, signal_handler)

# Arduino set up
ports = serial.tools.list_ports.comports()
ser = serial.Serial()

portList = []

for onePort in ports:
    portList.append(str(onePort))
    print(str(onePort))

portVar = '/dev/cu.usbserial-110'
ser.baudrate = 9600
ser.port = portVar
ser.open()


# Calibration for Aruco cameras
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

# Define cameras
# cap = cv.VideoCapture(1)    # Grip width
# cap3 = cv.VideoCapture(2)   # Bar Path
# cap2 = cv.VideoCapture(3)   # Blazepose


# Writing bar path data csv file
try:
    fp = open('out.csv', 'w')
except IOError:
    print('error')
    sys.exit(-1)

# mediapipe set up
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


def calculate_grip_width(cam1, record_request, grip_messages, image_messages):

    recording = False
    cap = cam1
    print("calculate_grip_width", cap.getBackendName(), file=sys.stderr)
    while not shutdown:

        ret, frame = cap.read()
        if ret:
            image_messages.put((time.time(), "grip", frame), block=False)

        try:
            request = record_request.get(block=False)
            # print(request)
            recording = request
        except queue.Empty:
            # print('no request')
            if not recording:
                continue
        if recording:

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
                if locB.shape[0] == 0 or locC.shape[0] == 0:
                    continue
                dist = np.linalg.norm(locB - locC)

                vB = np.squeeze(np.asarray(locB))
                vC = np.squeeze(np.asarray(locC))

                vector = vC - vB
                angle_rad = np.arctan2(vector[1], np.linalg.norm(vector[[0, 2]]))
                angle_deg = np.degrees(angle_rad)
                # print(angle_deg)

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
                    msg = (time.time(), dist, frame, angle_deg)  # Message is a tuple of (current time, grip width)
                    grip_messages.put(msg, block=False, timeout=None)
            # else:
            #     print('detected no markers', file=sys.stderr)

            # if ret:
            #     cv.imshow("frame", frame)


def calculate_blazepose(cam2, record_request2, pose_messages, image_messages):
    recording = False
    cap2 = cam2
    print("blazepose", cap2.getBackendName(), file=sys.stderr)

    # width = int(cap2.get(cv.CAP_PROP_FRAME_WIDTH))
    # height = int(cap2.get(cv.CAP_PROP_FRAME_HEIGHT))
    # writer = cv.VideoWriter('basicvideo3.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    while not shutdown:
        # Grab an image from the camera and send it off before anything else
        ret1, frame1 = cap2.read()
        if ret1:
            image_messages.put((time.time(), "blazepose", frame1), block=False)

        try:
            request = record_request2.get(block=False)
            # print(request)
            recording = request
        except queue.Empty:
            # print('no request')
            if not recording:
                continue

        if recording:
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
                msg_p = (time.time(), angle_l, angle_r, shoulder_distance)
                pose_messages.put(msg_p, block=False, timeout=None)

        # cv.VideoCapture(2).release()
        # writer.release()
        # cv.destroyAllWindows()


def calculate_barPath(cam3, record_request3, bar_messages):
    recording = False
    cap3 = cam3
    while not shutdown:
        try:
            request = record_request3.get(block=False)
            print(request)
            recording = request
        except queue.Empty:
            # print('no request')
            if not recording:
                continue
        if recording:
            ret3, frame3 = cap3.read()
            # if not ret3:
            #     break
            gray_frame = cv.cvtColor(frame3, cv.COLOR_BGR2GRAY)
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
                        frame3, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
                    )
                    corners = corners.reshape(4, 2)
                    corners = corners.astype(int)
                    top_right = corners[0].ravel()
                    top_left = corners[1].ravel()
                    bottom_right = corners[2].ravel()
                    bottom_left = corners[3].ravel()

                    # Draw the pose of the marker
                    point = cv.drawFrameAxes(frame3, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
                    msg_bar = (time.time(), locA)  # Message is a tuple of (current time, grip width)
                    bar_messages.put(msg_bar, block=False, timeout=None)


def calculate_arduino(arduino_messages):
    last_contact = UNRACKED
    first_time = True

    while not shutdown:
        if ser.in_waiting:
            # *THIS "ATTEMPT" ALWAYS SUCCEEDS BECAUSE IT WILL BLOCK UNTIL IT WORKS*
            try:
                packet = ser.readline()
                txt = packet.decode('ISO-8859-1').rstrip('\n').rstrip('\r')
                arrayTxt = txt.split(",")
                pressure = txt[0]

                contact = float(pressure)
                contact = int(contact)
                if first_time:
                    first_time = False
                    rack_state.put(contact)

                elif last_contact != UNRACKED and contact == UNRACKED:
                    print('Bar Racked')
                    rack_state.put(False)

                elif last_contact == UNRACKED and contact != UNRACKED:
                    print('Bar Unracked')
                    rack_state.put(True)

                last_contact = contact

            except IndexError:
                print('ignoring error')


def audio_player(audio_messages):

    while not shutdown:
        try:
            msg = audio_messages.get(block=False)
            filename = msg[0]
            current_time = msg[1]
            if time.time() - current_time < 0.001:
                playsound(filename)

        except queue.Empty:
            pass


audio_request = queue.Queue()
grip_record_request = queue.Queue()
pose_record_request = queue.Queue()
bar_record_request = queue.Queue()
grip_messages = queue.Queue()
pose_messages = queue.Queue()
bar_messages = queue.Queue()
image_messages = queue.Queue()
arduino_messages = queue.Queue()
rack_state = queue.Queue()

t0 = threading.Thread(target=calculate_arduino, args=(arduino_messages,))
t0.start()

t1 = threading.Thread(target=calculate_grip_width, args=(cv.VideoCapture(1), grip_record_request, grip_messages, image_messages))
t1.start()

t2 = threading.Thread(target=calculate_blazepose, args=(cv.VideoCapture(2), pose_record_request, pose_messages, image_messages))
t2.start()

# t3 = threading.Thread(target=calculate_barPath, args=(bar_messages,))
# t3.start()

t4 = threading.Thread(target=audio_player, args=(audio_request,))
t4.start()

grip_msg_received = 0
shoulder_msg_received = 0
grip_dist = 0
shoulder_dist = 0
left_angle_sum = 0
right_angle_sum = 0
left_angle_received = []
right_angle_received = []
left_angle_data = []
right_angle_data = []
bar_arr = []

wide = 'Grip width too wide'
tts = gTTS(text=wide, lang='en')
tts.save("wide.mp3")

narrow = 'Grip width too narrow'
tts = gTTS(text=narrow, lang='en')
tts.save("narrow.mp3")

okay = 'Grip width is within range. Please proceed.'
tts = gTTS(text=okay, lang='en')
tts.save("okay.mp3")


grip_record_request.put(True)
pose_record_request.put(True)

grip_dist_sum = 0
grip_msg_received = 0
shoulder_dist_sum = 0
shoulder_msg_received = 0
left_angle_sum = 0
right_angle_sum = 0
left_avg = 0
right_avg = 0
racked = True

width = 1920
height = 1080
video_writer = cv.VideoWriter('test.mp4', cv.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
images = {}

while not shutdown:

    try:
        racked = rack_state.get(block=False)
        print('racked = ', str(racked))

    except queue.Empty:
        pass

    try:
        grip_data = grip_messages.get(block=False)
        grip_dist = grip_data[1]
        grip_msg_received += 1
        bar_angle = grip_data[3]

    except queue.Empty:
        pass

    try:
        pose_data = pose_messages.get(block=False)
        left_angle_data.append(pose_data[1])
        right_angle_data.append(pose_data[2])
        shoulder_dist = pose_data[3]
        shoulder_msg_received += 1

    except queue.Empty:
        pass

    try:
        while not image_messages.empty():
            img_data = image_messages.get(block=False)

            if not img_data[1] in images.keys():
                images[img_data[1]] = [img_data[2]]
            else:
                images[img_data[1]].append(img_data[2])

    except queue.Empty:
        pass

    num_frames = [len(value) for value in images.values()]
    min_frames = min(num_frames)
    if min_frames > 0:
        for i in range(min_frames):
            frames = [value.pop(0) for key, value in images.items()]
            image = frames[0]
            for j in range(1, len(frames)):
                image = np.concatenate((image, frames[j]), axis=0)
            video_writer.write(image)


    # video_writer.flush()


    # try:
    #     bar_data = bar_messages.get(block=False)
    #     # print(bar_data[1])
    #     bar_arr.append(bar_data[1])

    # except queue.Empty:
    #     pass

    if racked:
        # may not think its racked
        # print('sending request')
        # grip_record_request.put(racked)
        # bar_record_request.put(racked)

        # tried while in the else statement and it works but keeps going after it is unracked
        # maybe change the inside if statements to while
        if grip_dist > 0 and shoulder_dist > 0:
            if grip_dist > shoulder_dist * 1.9:
                audio_request.put(("wide.mp3", time.time()))

            elif grip_dist < shoulder_dist * 1.1:
                audio_request.put(("narrow.mp3", time.time()))

            else:
                audio_request.put(("okay.mp3", time.time()))

            if abs(bar_angle) > 10:
                print('stop')

    else:
        if grip_msg_received > 0 and shoulder_msg_received > 0:

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

            # print(f'Left standard deviation: {left_std:.2f}')
            # print(f'Right standard deviation: {right_std:.2f}')

        grip_dist_sum = 0
        grip_msg_received = 0
        shoulder_dist_sum = 0
        shoulder_msg_received = 0
        left_angle_sum = 0
        right_angle_sum = 0
        left_avg = 0
        right_avg = 0

    if heardEnter():
        shutdown = True



print('released video')
video_writer.release()
print('released video')


t0.join()
t1.join()
t2.join()
# t3.join()
t4.join()
