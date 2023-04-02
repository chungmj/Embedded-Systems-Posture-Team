import cv2
import time

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution (modify this according to your camera capabilities)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        exit()

    # Initialize the time and frame counter
    start_time = time.time()
    frame_counter = 0

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to read the frame.")
                break
            
            frame_counter += 1

            # Calculate frame rate
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_counter / elapsed_time
                fps_text = "FPS: {:.2f}".format(fps)

                # Put the frame rate text on the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_color = (255, 0, 255)
                font_thickness = 2
                cv2.putText(frame, fps_text, (10, 30), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # Display the captured frame with the frame rate
            cv2.imshow('Camera', frame)

            # Exit the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
