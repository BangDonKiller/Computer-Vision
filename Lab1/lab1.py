### imoprt library ###
import numpy as np
import cv2

### Capture from camera or Read an video ###
cap = cv2.VideoCapture("./Lab1/minion_video.avi")

### Display the frame ###
while cap.isOpened():
    ret, frame = cap.read()

    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]

    # copy frame
    temp_frame = np.zeros_like(frame)

    cv2.imshow("110502510_ShiehMingYu_original_frame", frame)

    ### Do the processing (convert RGB to grayscale)###
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("110502510_ShiehMingYu_grey_frame", gray)

    if cv2.waitKey(1) & 0xFF == ord("r"):
        red = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])
        cv2.imwrite("Lab1/110502510_ShiehMingYu_Capture_r.png", red)
        cv2.imshow("red", red)

    elif cv2.waitKey(1) & 0xFF == ord("g"):
        green = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
        cv2.imwrite("Lab1/110502510_ShiehMingYu_Capture_g.png", green)
        cv2.imshow("green", green)

    elif cv2.waitKey(1) & 0xFF == ord("b"):
        blue = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])
        cv2.imwrite("Lab1/110502510_ShiehMingYu_Capture_b.png", blue)
        cv2.imshow("blue", blue)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

### Close and Exit ###
cap.release()
cv2.destroyAllWindows()
