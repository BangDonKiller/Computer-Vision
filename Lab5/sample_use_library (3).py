import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread("./Lab5/littleMINI (1).jpg", cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
laplacian = np.uint8(np.absolute(laplacian))

sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelX = np.uint8(np.absolute(sobelX))

sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobelY = np.uint8(np.absolute(sobelY))

canny = cv2.Canny(img, 100, 200)

title = ["image", "Laplacian", "sobelX", "sobelY", "Canny"]
image = [img, laplacian, sobelX, sobelY, canny]
for i in range(5):
    plt.subplot(2, 3, i + 1), plt.imshow(image[i], "gray")
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])
plt.show()
