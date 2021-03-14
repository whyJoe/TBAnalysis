# -*-coding:utf-8-*-
import cv2 as cv
import numpy as np


# 追踪物体
def extrace_object_demo():
    capture = cv.VideoCapture("")
    while (True):
        ret, frame = capture.read()
        if ret == False:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([37, 43, 46])
        upper_hsv = np.array([77, 255, 255])
        mask = cv.inRange(frame, lowerb=lower_hsv, upperb=upper_hsv)
        cv.imshow("video", frame)
        c = cv.waitKey(40)
        if c == 27:
            break


# 色彩空间转换
def color_space_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("HSV", HSV)
    YUV = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow("YUV", YUV)
    YCRCB = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)
    cv.imshow("YCRCB", YCRCB)


src = cv.imread("test.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
color_space_demo(src)

b, g, r = cv.split(src)
cv.imshow("blue",b)
cv.imshow("green",g)
cv.imshow("red",r)
cv.imshow("image",src)
src[:,:,2] = 0
src = cv.merge([b, g, r])
cv.imshow("changed image",src)

cv.waitKey(0)
cv.destroyAllWindows()
