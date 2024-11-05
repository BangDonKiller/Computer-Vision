import cv2
import numpy as np

# read an RGB image to Gray level
image = cv2.imread("lenna (1).png", cv2.IMREAD_GRAYSCALE)

### create two masks, including Horizontal edges and Vertical edges of Sobel ###
sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

### Calculate image_sobelx and image_sobely using masks ###
image_sobelx = cv2.filter2D(image, -1, sobel_x_kernel)
image_sobely = cv2.filter2D(image, -1, sobel_y_kernel)

# Calculate an approximation of the gradient: G=sqrt(Gx*Gx+Gy*Gy)
sobelGrad = np.sqrt(
    image_sobelx.astype(np.float32) ** 2 + image_sobely.astype(np.float32) ** 2
).astype(np.uint8)

# Although sometimes the following simpler equation: G=|Gx|+|Gy|
sobelGrad2 = (np.abs(image_sobelx) + np.abs(image_sobely)).astype(np.uint8)
sobelCombined2 = cv2.bitwise_or(np.abs(image_sobelx), np.abs(image_sobely)).astype(
    np.uint8
)

# show image #
# cv2.imshow("image_gray", image)
cv2.imshow("image_sobelx", image_sobelx.astype(np.uint8))
cv2.imshow("image_sobely", image_sobely.astype(np.uint8))
cv2.imshow("image_sobelGrad", sobelGrad)
cv2.imshow("image_sobelCombined2", sobelCombined2)
print("equal: ", not (np.bitwise_xor(sobelGrad, sobelCombined2).any()))
cv2.waitKey(0)
cv2.destroyAllWindows()
