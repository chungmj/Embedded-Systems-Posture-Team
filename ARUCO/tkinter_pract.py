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
import PySimpleGUI as sg

sg.theme('DarkGrey13')

# Define the GUI layout
layout = [[sg.Text('Queueing and Multithreading with PySimpleGUI')],
          # [sg.Text('Enter the number of tasks:'), sg.Input(key='-NUM_TASKS-', size=(10, 1))],
          [sg.Button('Start'), sg.Exit()],
          [sg.Multiline(size=(60, 20), key='-MULTILINE-')]]

# Create the window
window = sg.Window('Queueing and Multithreading with PySimpleGUI', layout)

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

portVar = '/dev/cu.usbserial-140'
ser.baudrate = 9600
ser.port = portVar
ser.open()

UNRACKED = 5.0

# mediapipe set up
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # hip
    b = np.array(b)  # shoulder
    c = np.array(c)  # elbow

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_blazepose(cam2, record_request2, pose_messages):
    recording = False
    cap2 = cam2
    print("blazepose", cap2.getBackendName(), file=sys.stderr)

    while not shutdown:
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
                # window["-OUTPUT-"].update(shoulder_distance)


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
                roll_rad = math.atan2(float(arrayTxt[1]), float(arrayTxt[2]))
                roll_angle = math.degrees(roll_rad)
                roll_angle_arr.append(roll_angle)

                contact = float(arrayTxt[3])
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

                # elif last_contact != UNRACKED and contact != UNRACKED:
                #     rack_state.put(False)
                # if last_contact > 2 and contact > 2:
                #     print('Bar Racked')
                #     rack_state.put(False)
                # elif last_contact < 2 and contact > 2:
                #     print('Bar Unracked')
                #     rack_state.put(True)
                last_contact = contact

            except IndexError:
                print('ignoring error')


pose_record_request = queue.Queue()
pose_messages = queue.Queue()
arduino_messages = queue.Queue()
rack_state = queue.Queue()


# t2 = threading.Thread(target=calculate_blazepose, args=(cv.VideoCapture(0), pose_record_request, pose_messages))
# t2.start()

shoulder_msg_received = 0
shoulder_dist = 0
left_angle_sum = 0
right_angle_sum = 0
left_angle_received = []
right_angle_received = []
left_angle_data = []
right_angle_data = []
roll_angle_arr = []
roll_count = 0
racked = True

pose_record_request.put(True)
shoulder_dist_sum = 0
shoulder_msg_received = 0
left_angle_sum = 0
right_angle_sum = 0
left_avg = 0
right_avg = 0

t0 = threading.Thread(target=calculate_arduino, args=(arduino_messages,))
t0.start()

t2 = threading.Thread(target=calculate_blazepose,
                      args=(cv.VideoCapture(0), pose_record_request, pose_messages))
t2.start()

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Start':
        while True:
            # t0 = threading.Thread(target=calculate_arduino, args=(arduino_messages,))
            # t0.start()
            #
            # t2 = threading.Thread(target=calculate_blazepose,
            #                       args=(cv.VideoCapture(0), pose_record_request, pose_messages))
            # t2.start()
            try:
                racked = rack_state.get(block=False)
                # print('racked = ', str(racked))

            except queue.Empty:
                pass

            try:
                pose_data = pose_messages.get(block=False)
                left_angle_data.append(pose_data[1])
                right_angle_data.append(pose_data[2])
                shoulder_dist = pose_data[3]
                shoulder_msg_received += 1


                # sg.Print(f'Shoulder distance: {pose_data[3]:.2f} cm')
                # text = f'{pose_data[3]}\n{pose_data[3]}'
                #
                # window["-MULTILINE-"].update(text)

            except queue.Empty:
                pass

            if racked:
                sg.Print('racked')
            else:
                if shoulder_msg_received > 0:
                    sg.Print(f'Shoulder distance: {pose_data[3]:.2f} cm')

                grip_dist_sum = 0
                grip_msg_received = 0
                shoulder_dist_sum = 0
                shoulder_msg_received = 0
                left_angle_sum = 0
                right_angle_sum = 0
                left_avg = 0
                right_avg = 0
                roll_count = 0

t0.join()
t2.join()
