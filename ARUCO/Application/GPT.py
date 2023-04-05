import multiprocessing as mp

import cv2
def capture_video(video_fd, frame_queue):
    print("trying to open")
    cap = cv2.VideoCapture(video_fd)
    print("open")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    cap.release()
    frame_queue.put(None)  # Signal that the capture process is done

def main():
    camera_index = 0  # Use 0 for the default camera or change the index if you have multiple cameras
    # cap = cv2.VideoCapture(camera_index)
    # video_fd = int(cap.get(cv2.CAP_PROP_FOURCC))
    # cap.release()

    frame_queue = mp.Queue(maxsize=10)  # Adjust maxsize as needed

    capture_process = mp.Process(target=capture_video, args=(camera_index, frame_queue))
    capture_process.start()

    while True:
        frame = frame_queue.get()
        if frame is None:  # Capture process is done
            break

        # Process the frame in the main process as needed
        cv2.imshow("Frame", frame)
        # print(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture_process.join()
    cv2.destroyAllWindows()

main()
