# -*-coding:utf-8-*-
import cv2 as cv
import numpy as np


# 遍历图片像素点
def access_pixels(image):
    print(image.shape)
    heigth = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("width : %s,heigth : %s,channels : %s" % (width, heigth, channels))
    # 循环每个像素
    for row in range(heigth):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv
    cv.imshow("pixels_demo", image)


# 构建一张图片
def create_image():
    # """多通道"""
    # img = np.zeros([400, 400, 3], np.uint8)
    # img[:, :, 0] = np.ones([400, 400]) * 127
    # cv.imshow("image", img)

    # """单通道"""
    # img = np.zeros([400, 400, 1], np.uint8)
    # img = img * 0
    # cv.imshow("new image",img)
    # cv.imwrite("D://result.png",img)

    # 构建数组,若字符类型表示范围不够大则会截断
    m1 = np.ones([3, 3], np.uint8)
    m1.fill(12222.388)
    print(m1)

    # 降维，转化为一行九列
    m2 = m1.reshape([1, 9])
    print(m2)


print("------------Hello python------------")
# src = cv.imread("timg.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
t1 = cv.getTickCount()

create_image()
# access_pixels(src)

t2 = cv.getTickCount()
time = (t2 - t1) / cv.getTickFrequency()
print("time : %s ms" % time * 1000)
cv.waitKey(0)
cv.destroyAllWindows()
