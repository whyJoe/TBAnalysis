# -*-coding:utf-8-*-
import cv2
import numpy as np

"""
高斯模糊/噪声
轮廓还在，保留图像的主要特征
高斯模糊比均值模糊去噪效果好
"""


def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv


def gaussion_noise(image):
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv2.imshow("noise image", image)


if __name__ == "__main__":
    src = cv2.imread("test.jpg")  # blue green red
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("image", src)
    print(src)
    gaussion_noise(src)
    # 若ksize不为(0, 0)，则按照ksize计算，后面的sigmaX没有意义。若ksize为(0, 0)，则根据后面的sigmaX计算ksize
    gaussian = cv2.GaussianBlur(src, (5, 5), 0)  # 高斯模糊
    cv2.imshow("gaussian", gaussian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
