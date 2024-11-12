import numpy as np
import cv2


## 檢查機制
def img_equal(src1, src2):
    return not (np.bitwise_xor(src1, src2).any())


## 膨脹運算
def dilate_(src, kernel, iterations=1):
    # Patch Index-List
    rowIndexList = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    colIndexList = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    #
    img = src.copy()
    for iters in range(iterations):
        ###################################################################################################complete zone
        img = np.pad(
            img, ((1, 1), (1, 1)), "constant", constant_values=((0, 0), (0, 0))
        )  # padding ,pad 0
        dilation_output = np.zeros(img.shape, dtype=np.uint8)  # Blank space
        for center_row in range(1, img.shape[0] - 1):
            for center_col in range(1, img.shape[1] - 1):
                #! Get patch (3 x 3)
                patch = img[rowIndexList + center_row, colIndexList + center_col]
                #! Get mul : (elementwise-mutiple of array)
                mul = np.multiply(patch, kernel)
                #! Condition : if mul have any 255,than set center-pixel = 255, other = 0
                if (mul == 255).any():
                    dilation_output[center_row, center_col] = 255

        img = dilation_output[1:-1, 1:-1]
        # --------------------------------------------------------------------------------------------------complete zone
    return img


## 侵蝕運算
def erode_(src, kernel, iterations=1):
    # IndexList
    rowIndexList = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    colIndexList = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    #
    img = src.copy()
    for iters in range(iterations):
        ###################################################################################################complete zone
        img = np.pad(
            img, ((1, 1), (1, 1)), "constant", constant_values=((0, 0), (0, 0))
        )  # padding,pad 0
        erosion_output = np.zeros(img.shape, dtype=np.uint8)  # Blank space
        for center_row in range(1, img.shape[0] - 1):
            for center_col in range(1, img.shape[1] - 1):
                #! Get patch (3 x 3)
                patch = img[rowIndexList + center_row, colIndexList + center_col]

                #! Get mul : mul = patch * kernel (elementwise-mutiply)
                mul = np.multiply(patch, kernel)

                #! Condition : if mul and kernel are equal, than center pixel = 255, other = 0
                erosion_output[center_row, center_col] = (
                    255 if np.all(mul[kernel == 1] != 0) else 0
                )

        img = erosion_output[1:-1, 1:-1]
        # --------------------------------------------------------------------------------------------------complete zone
    return img


def open_(src, kernel, iterations=1):
    ###################################################################################################complete zone
    #!
    erosion = erode_(src, kernel, iterations=iterations)
    opening = dilate_(erosion, kernel, iterations=iterations)
    return opening
    # --------------------------------------------------------------------------------------------------complete zone


def close_(src, kernel, iterations=1):
    ###################################################################################################complete zone
    #!
    dilation = dilate_(src, kernel, iterations=iterations)
    closing = erode_(dilation, kernel, iterations=iterations)
    return closing
    # --------------------------------------------------------------------------------------------------complete zone


if __name__ == "__main__":
    ## Read Image
    source_img = cv2.imread("j.png", 0)  # cv2.imread("j.png", 0)

    open_img = cv2.imread("opening_j.png", 0)

    close_img = cv2.imread("closing_j.png", 0)

    ## Set Kernel
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    ## Your Version
    dilation = dilate_(source_img, kernel, iterations=1)
    erosion = erode_(source_img, kernel, iterations=1)
    opening = open_(open_img, kernel, iterations=2)
    closing = close_(close_img, kernel, iterations=2)

    ## Opencv Version Anwser
    dilation_ans = cv2.dilate(source_img, kernel, iterations=1)
    erosion_ans = cv2.erode(source_img, kernel, iterations=1)
    opening_ans = cv2.morphologyEx(open_img, cv2.MORPH_OPEN, kernel, iterations=2)
    closing_ans = cv2.morphologyEx(close_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    ## Your Version Display
    cv2.imshow("dilate_answer", dilation)
    cv2.imshow("erode_answer", erosion)
    cv2.imshow("opening_answer", opening)
    cv2.imshow("closing_answer", closing)

    ## Opencv Version Anwser Display
    cv2.imshow("answer_dilation", dilation_ans)
    cv2.imshow("answer_erosion", erosion_ans)
    cv2.imshow("answer_opening", opening_ans)
    cv2.imshow("answer_closing", closing_ans)

    ## Check Answer
    print("dilate Check:\t", img_equal(dilation, dilation_ans))
    print("erosion Check:\t", img_equal(erosion, erosion_ans))
    print("opening Check:\t", img_equal(opening, opening_ans))
    print("closing Check:\t", img_equal(closing, closing_ans))

    cv2.waitKey(0)
