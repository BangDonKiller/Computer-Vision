import cv2
import numpy as np
import os


def BGR2GRAY2ThreeChannel(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_gray


def negativeLinear(img):
    img = 255 - img
    img = img.astype(np.uint8)
    return img


def logTransformation(img):
    c = 255 / np.max(np.log10(1.0 + img))
    img = c * np.log10(1.0 + img)
    img = img.astype(np.uint8)
    return img


def segmentation_And_processing(img):
    img_up_left, img_up_right, img_down_left, img_down_right = (
        img[0 : img.shape[0] // 2, 0 : img.shape[1] // 2],
        img[0 : img.shape[0] // 2, img.shape[1] // 2 : img.shape[1]],
        img[img.shape[0] // 2 : img.shape[0], 0 : img.shape[1] // 2],
        img[img.shape[0] // 2 : img.shape[0], img.shape[1] // 2 : img.shape[1]],
    )

    img_up_left = negativeLinear(img_up_left)
    img_down_left = negativeLinear(img_down_left)
    img_down_left = BGR2GRAY2ThreeChannel(img_down_left)

    img_up_right = BGR2GRAY2ThreeChannel(img_up_right)
    img_up_right = logTransformation(img_up_right)
    img_down_right = logTransformation(img_down_right)

    return img_up_left, img_up_right, img_down_left, img_down_right


def segmentConcatenate(img_up_a, img_up_b, img_down_a, img_down_b):
    # img_up_b 新增第二維度
    img_up_b = np.expand_dims(img_up_b, axis=2)
    img_up_b = cv2.cvtColor(img_up_b, cv2.COLOR_GRAY2BGR)
    concate_up = np.concatenate((img_up_a, img_up_b), axis=1)

    # img_down_b 新增第二維度
    img_down_a = np.expand_dims(img_down_a, axis=2)
    img_down_a = cv2.cvtColor(img_down_a, cv2.COLOR_GRAY2BGR)
    concate_down = np.concatenate((img_down_a, img_down_b), axis=1)

    concate = np.concatenate((concate_up, concate_down), axis=0)

    return concate


def imgShow(studentID, img):
    cv2.imshow(studentID, img)
    cv2.imwrite(studentID + ".jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img = cv2.imread(f"{os.path.dirname(__file__)}/1131_Lab2_Enhancement.jpg")

    img_bgr = cv2.cvtColor(img, cv2.IMREAD_COLOR)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    img_output = logTransformation(img_bgr)
    imgShow("student_id_lab2", img_output)

    img_output = negativeLinear(img_bgr)
    imgShow("student_id_lab2", img_output)

    img_output = logTransformation(img_gray)
    imgShow("student_id_lab2", img_output)

    img_output = negativeLinear(img_gray)
    imgShow("student_id_lab2", img_output)

    img_up_left, img_up_right, img_down_left, img_down_right = (
        segmentation_And_processing(img)
    )

    img_output = segmentConcatenate(
        img_up_left, img_up_right, img_down_left, img_down_right
    )

    imgShow("110502510_lab2", img_output)


if __name__ == "__main__":
    main()
