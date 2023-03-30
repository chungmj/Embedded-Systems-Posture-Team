# import the opencv library
import cv2

# define a video capture object
vid1 = cv2.VideoCapture(0)
vid2 = cv2.VideoCapture(2)
vid3 = cv2.VideoCapture(3)

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid1.read()
    print(ret)
    ret2, frame2 = vid2.read()
    print(ret2)
    ret3, frame3 = vid3.read()
    print(ret3)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('frame2', frame2)
    cv2.imshow('frame3', frame3)


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid1.release()
vid2.release()
vid3.release()

# Destroy all the windows
cv2.destroyAllWindows()

# https://stackoverflow.com/questions/49663474/opencv-python-cv2-videocapture-can-only-find-2-of-3-cameras-windows-camera-app
# ID test
# cams_test = 500
# for i in range(0, cams_test):
#     cap = cv2.VideoCapture(i)
#     test, frame = cap.read()
#     if test:
#         print("i : "+str(i)+" /// result: "+str(test))

