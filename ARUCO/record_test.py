import cv2


cap = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


writer = cv2.VideoWriter('basicvideo2.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))
writer2 = cv2.VideoWriter('basicvideo2_back.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))


while True:
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    writer.write(frame)
    writer2.write(frame2)

    cv2.imshow('frame', frame)
    cv2.imshow('frame', frame2)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
writer.release()
cv2.destroyAllWindows()
