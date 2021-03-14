# -*-coding:utf-8-*-
import cv2 as cv
import numpy as np


# 均值模糊
def blur_demo(image):
    # x和y方向上的模糊
    dst = cv.blur(image, (5, 5))
    cv.imshow("blur_demo", dst)


# 中值模糊
def median_blur_demo(image):
    dst = cv.medianBlur(image, 5)
    cv.imshow("median_blur_demo", dst)


# 用户自定义模糊，可以调整图片锐化度，锐化算子总和为1或0，最好是奇数
def custom_blur_demo(image):
    # kernel = np.ones([5, 5], np.float32) / 25
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst = cv.filter2D(image, -1, kernel=kernel)
    cv.imshow("custom_blur_demo", dst)


print("----------hello----------")
src = cv.imread("test.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
custom_blur_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
