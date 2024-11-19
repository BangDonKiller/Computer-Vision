import cv2
import numpy as np
from math import atan2, sqrt, pi

# read image
grayImage = cv2.imread("./Lab7/olaf.jpg", cv2.IMREAD_GRAYSCALE)

# print(grayImage)

## convert to binary image
#! Syntax: cv2.threshold(img, thresh, maxval, type)
ret, binaryImage = cv2.threshold(grayImage, 250, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("Binary image", binaryImage)

## morphological
#! Syntax: cv2.morphologyEx(img, op, kernel)，Yoy can set op to cv2.MORPH_OPEN
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel)

cv2.imshow("Morphological", opening)

# find and draw contours
# cv2.findContours()，The input image can only be a binary image
contours, hierarchy = cv2.findContours(
    opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
pca_ellipse = cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB)
cv2.drawContours(pca_ellipse, contours, -1, (0, 0, 255), 3)
cv2.imshow("Contours", pca_ellipse)


## PCA analysis
def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]  # x coordinate
        data_pts[i, 1] = pts[i, 0, 1]  # y coordinate

    # Perform PCA analysis

    #!Syntax: cv2.PCACompute2()--> mean, eigenvalues, eigenvectors
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, np.empty((0)))

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    ## Compute orientation angle， convert from radians to degrees by multiply 180/pi
    #!Syntax: atan2()
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / pi

    ## Compute lengths of the major and minor axes
    #!major_axis_length=
    #!minor_axis_length=
    major_axis_length = sqrt(eigenvalues[0, 0])
    minor_axis_length = sqrt(eigenvalues[1, 0])

    return cntr, angle, (major_axis_length, minor_axis_length)


# center of the object, rotational angle, length of the major axis and minor axis
for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv2.contourArea(c)
    # Ignore contours that are too small or too large
    if area < 1e2 or 1e5 < area:
        continue
    # Draw each contour only for visualisation purposes
    # Find the orientation of each shape
    cntr, angle, (major_axis_length, minor_axis_length) = getOrientation(c, pca_ellipse)

    ## draw center circle and ellipse
    #! Syntax: cv2.circle(image, centerCoordinates, radius, color[, thickness[, lineType[, shift]]])
    #! Syntax: cv2.ellipse(image, centerCoordinates, axesLength, angle, startAngle, endAngle, color [, thickness[, lineType[, shift]]])
    cv2.circle(pca_ellipse, cntr, 3, (255, 0, 255), 3)
    cv2.ellipse(
        pca_ellipse,
        cntr,
        (int(major_axis_length * 2), int(minor_axis_length * 2)),
        angle,
        0,
        360,
        (0, 255, 0),
        2,
    )


## show images and write image
cv2.imshow("Gray image", grayImage)
cv2.imshow("Binary image", binaryImage)
cv2.imshow("Morphological", opening)
cv2.imshow("PCA ellipse", pca_ellipse)
cv2.imwrite("PCA_ellipse.jpg", pca_ellipse)
cv2.waitKey(0)
cv2.destroyAllWindows()
