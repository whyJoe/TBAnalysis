# -*-coding:utf-8-*-
import cv2 as cv
import numpy as np


# 高斯双边模糊
def bi_demo(image):
    dst = cv.bilateralFilter(image, 0, 50, 20)
    cv.imshow("bilateralFilter", dst)


# 均值迁移
def shift_demo(image):
    dst = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow("shift_demo", dst)


print("----------hello----------")
src = cv.imread("szh1.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
print(src)
cv.imshow("input image", src)
bi_demo(src)
# print("==>",src)
# shift_demo(src)
# print("=========>",src)
cv.waitKey(0)
cv.destroyAllWindows()
