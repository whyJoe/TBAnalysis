# -*-coding:utf-8-*-
import cv2 as cv
import numpy as np


print("----------hello----------")
src = cv.imread("timg.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
cv.waitKey(0)
cv.destroyAllWindows()
