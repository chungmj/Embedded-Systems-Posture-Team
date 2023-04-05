"""
Author: Gym Sense
Date: 4/4/2023
"""

import os
import signal
import time
# import cv2 as cv
from multiprocessing import Process, Queue, set_start_method

if __name__ == "__main__":
    try:
        set_start_method("spawn")
        print("Fixed camera issues!")
    except Exception:
        print("Pray")

    import Arduino
    import Aruco
    import Pose
    import Audio
    import Display

    arduino_messages = Queue()

    grip_messages        = Queue()
    aruco_image_messages = Queue()

    pose_messages       = Queue()
    pose_image_messages = Queue()

    audio_requests = Queue()

def setup():
    """
    Initalizes resources for the program.
    """
    # Open Cameras
    side_cam     = 2
    overhead_cam = 0


    processes = []
    processes.append( Process(target=Arduino.calculate_arduino, args=(arduino_messages,)) )

    processes.append( Process(target=Aruco.calculate_grip_width, \
                args=(side_cam, grip_messages, aruco_image_messages)) )

    processes.append( Process(target=Pose.calculate_blazepose, \
                args=(overhead_cam, pose_messages, pose_image_messages)) )

    processes.append( Process(target=Audio.audio_player, \
                args=(audio_requests,)) )
    
    for p in processes:
        p.start()

    return processes

def read_arduino(display):
    for i in range(4):
        if not arduino_messages.empty():
            msg = arduino_messages.get(block=False)
            # print(msg)
            if msg[0] == "BAR":
                display.update_var("Racked", msg[1])
            elif msg[0] == "BENCH":
                display.update_var("Arched", msg[1])
                display.update()


def read_aruco(display):
    if not grip_messages.empty():
        t, width, tilt = grip_messages.get()
        display.update_var("Grip Width", width)
        display.update_var("Bar Tilt", tilt)
    
    if not aruco_image_messages.empty():
        msg = aruco_image_messages.get()
        # Time, "grip", Raw, Annotated
        if len(msg) == 4:
            display.update_aruco_frames(msg[2], msg[3])
        else:
            display.update_aruco_frames(msg[2], None)


def read_pose(display):
    if not pose_messages.empty():
        t, angle_l, angle_r, shoulder_dist = pose_messages.get()
        display.update_var("Left Elbow Angle", angle_l)
        display.update_var("Right Elbow Angle", angle_r)
        display.update_var("Shoulder Width", shoulder_dist)

    if not pose_image_messages.empty():
        msg = pose_image_messages.get()
        # Time, "blazepose", Raw
        display.update_pose_frames(msg[2], None)

def main():
    processes = setup()
    display = Display.RealTimeApplication()

    try:
        with open(f"Partcipant{int(time.time())}.csv", "w") as csv_writer:
            first_time = True
            while True:
                read_arduino(display)
                read_pose(display)
                read_aruco(display)

                if first_time:
                    for key in display.variables.keys():
                        csv_writer.write(f"{key},")
                    csv_writer.write("\n")
                    first_time = False
                
                if not display.get_var("Racked"):
                    for key, value in display.variables.items():
                        csv_writer.write(f"{value.get()},")
                    csv_writer.write("\n")
                
                display.update()
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass

    for p in processes:
        os.kill(p.pid, signal.SIGINT)
        p.join()
    
if __name__ == "__main__":
    main()
