### imoprt library ###
import numpy as np
import cv2

### Capture from camera or Read an video ###
cap = cv2.VideoCapture('./Lab1/CloudFormationVideo.avi')

### Display the frame ###
while(cap.isOpened()):
    ret, frame = cap.read()
    ### Do the processing (convert RGB to grayscale)###
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

### Close and Exit ###
cap.release()
cv2.destroyAllWindows()


