# -*-coding:utf-8-*-
import cv2 as cv
import numpy as np

"""
两张图片之间的加减乘除
需两张图片尺寸相等    
"""


def add_demo(m1, m2):
    src = cv.add(m1, m2)
    cv.imshow("add_demo", src)


def subtract_demo(m1, m2):
    src = cv.subtract(m1, m2)
    cv.imshow("subtract_demo", src)


def divide_demo(m1, m2):
    src = cv.divide(m1, m2)
    cv.imshow("divide_demo", src)


def multiply_demo(m1, m2):
    src = cv.multiply(m1, m2)
    cv.imshow("multiply_demo", src)


# 求均值
def means_demo(m1, m2):
    M1, dev1 = cv.meanStdDev(m1)
    M2, dev2 = cv.meanStdDev(m2)
    print("m1均值: %s \n m1方差: %s" % (M1, dev1))
    print("m2均值: %s \n m2方差: %s" % (M2, dev2))
    # h, w = m1.shape[:2]
    # img = np.zeros([h, w], dtype=np.uint8)
    # print(img)


"""
    逻辑运算
"""


def logic_demo(m1, m2):
    M1 = cv.bitwise_and(m1, m2)
    M2 = cv.bitwise_or(m1, m2)
    M3 = cv.bitwise_not(m1, m2)
    M4 = cv.bitwise_xor(m1, m2)
    cv.imread()


def contrast_brightness_demo(image,c,b):
    h,w,ch = image.shape
    blank = np.zeros([h,w,ch],image.dtype)
    """
    src1 – First source array.
    alpha – Weight for the first array elements.
    src2 – Second source array of the same size and channel number as src1 .
    beta – Weight for the second array elements.
    dst – Destination array that has the same size and number of channels as the input arrays.
    gamma – Scalar added to each sum.
    dtype – Optional depth of the destination array. When both input arrays have the same depth
    """
    """c:对比度  b:亮度"""
    dst = cv.addWeighted(image,c,blank,1-c,b)
    cv.imshow("con_bri_demo",dst)


m1 = cv.imread(r"D:\tool\OpenCV\opencv\sources\samples\data\apple.jpg")
m2 = cv.imread("WindowsLogo.jpg")
# print(m1.shape)
# print(m2.shape)
# add_demo(m1,m2)
# subtract_demo(m1,m2)
# divide_demo(m1,m2)
# multiply_demo(m1, m2)
cv.imshow("image",m1)
contrast_brightness_demo(m1,2,0)
# means_demo(m1, m2)
cv.waitKey(0)
cv.destroyAllWindows()
