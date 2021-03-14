# -*-coding:utf-8-*-
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 构造直方图
def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()

# 彩色图像的直方图api
def image_hist(image):
    color = ('blue','green','red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image],[i],None,[256],[0,256])
        plt.plot(hist,color=color)
        plt.xlim([0,256])
    plt.show()


# print("----------hello----------")
# src = cv.imread("szh1.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
# # plot_demo(src)
# image_hist(src)
# cv.waitKey(0)
# cv.destroyAllWindows()
a = np.zeros(5)
print(a)