# -*-coding:utf-8-*-
import cv2 as cv
import numpy as np


# 直方图均衡化(默认)
def equalHist_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    cv.imshow("equalHist_demo", dst)


# 局部直方图均衡化(局部自适应)
def clahe_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    cv.imshow("clahe_image", dst)

#
def s():
    gg = np.zero([2*2*2, 1], np.int8)
    gg[2,0] = 5
    # gg[]
    print(gg)


print("----------hello----------")
src = cv.imread("test.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
# equalHist_demo(src)
# clahe_demo(src)
s()
cv.waitKey(0)
cv.destroyAllWindows()
