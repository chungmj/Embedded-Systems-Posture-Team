# import the opencv library
import cv2

# define a video capture object
vid1 = cv2.VideoCapture(1)
vid2 = cv2.VideoCapture(2)
vid3 = cv2.VideoCapture(3)

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid1.read()
    ret2, frame2 = vid2.read()
    ret3, frame3 = vid3.read()
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

