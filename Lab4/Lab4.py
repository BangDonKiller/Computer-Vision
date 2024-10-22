import numpy as np
import cv2


def apply_skin_color_segmentation_hsv(frame):
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define skin color range for HSV
    lower1 = np.array([170, 10, 60], np.uint8)
    upper1 = np.array([180, 225, 225], np.uint8)
    lower2 = np.array([0, 10, 60], np.uint8)
    upper2 = np.array([20, 225, 225], np.uint8)

    # 將兩個遮罩取or運算
    skin1 = cv2.inRange(hsv_frame, lower1, upper1)
    skin2 = cv2.inRange(hsv_frame, lower2, upper2)
    hsv_mask = cv2.bitwise_or(skin1, skin2)

    # Apply opening to HSV mask to remove some noise
    kernel = np.ones((5, 5), np.uint8)
    hsv_mask = cv2.erode(hsv_mask, kernel, iterations=2)
    hsv_mask = cv2.dilate(hsv_mask, kernel, iterations=2)

    return hsv_mask


def apply_skin_color_segmentation_ycrcb(frame):
    # Convert the frame to YCrCb color space
    ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Define skin color range for YCrCb
    ycrcb_min = np.array([80, 133, 77], np.uint8)
    ycrcb_max = np.array([255, 177, 127], np.uint8)
    ycrcb_mask = cv2.inRange(ycrcb_frame, ycrcb_min, ycrcb_max)

    # Apply opening to YCrCb mask to remove some noise
    kernel = np.ones((3, 3), np.uint8)
    ycrcb_mask = cv2.erode(ycrcb_mask, kernel, iterations=1)
    ycrcb_mask = cv2.dilate(ycrcb_mask, kernel, iterations=1)

    return ycrcb_mask


def main():
    cap = cv2.VideoCapture("Lab4.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Apply gaussian blur to the frame
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 2)

        # Apply skin color segmentation for YCrCb and HSV
        ycrcb_mask = apply_skin_color_segmentation_ycrcb(blurred_frame)
        hsv_mask = apply_skin_color_segmentation_hsv(blurred_frame)

        # Find contours for the YCrCb and HSV masks
        contours_ycrcb, _ = cv2.findContours(
            ycrcb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_hsv, _ = cv2.findContours(
            hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw contours on the original frame for both masks
        ycrcb_contoured = blurred_frame.copy()
        hsv_contoured = blurred_frame.copy()
        cv2.drawContours(ycrcb_contoured, contours_ycrcb, -1, (0, 255, 0), 2)
        cv2.drawContours(hsv_contoured, contours_hsv, -1, (0, 255, 0), 2)

        # Apply the mask to the original frame for visualization
        ycrcb_masked = cv2.bitwise_and(blurred_frame, blurred_frame, mask=ycrcb_mask)
        hsv_masked = cv2.bitwise_and(blurred_frame, blurred_frame, mask=hsv_mask)

        # 將mask上輪廓
        cv2.drawContours(ycrcb_masked, contours_ycrcb, -1, (0, 255, 0), 2)
        cv2.drawContours(hsv_masked, contours_hsv, -1, (0, 255, 0), 2)

        # Display the results in separate windows
        cv2.imshow("YCrCb Output", ycrcb_contoured)
        cv2.imshow("YCrCb Output Masked", ycrcb_masked)
        cv2.imshow("HSV Output", hsv_contoured)
        cv2.imshow("HSV Output Masked", hsv_masked)

        # Break loop on 'q' or 'Q' key press, otherwise proceed with the next frame
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # gpg test
    main()
