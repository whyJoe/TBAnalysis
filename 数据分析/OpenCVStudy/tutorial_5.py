# -*-coding:utf-8-*-
import cv2 as cv
import numpy as np


# mask必须加2
def fill_color_demo(image):
    copyImg = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    # floodFill( 1.操作的图像, 2.掩模, 3.起始像素值，4.填充的颜色, 5.填充颜色的低值， 6.填充颜色的高值 ,7.填充的方法)
    cv.floodFill(copyImg, mask, (50,50), (0, 255, 255), (50, 50, 50), (30, 30, 30), cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("fill_color_demo", copyImg)


# mask必须加2，设置矩阵为1
def fill_color_binary():
    image = np.zeros([400, 400, 3], np.uint8)
    image[50:300, 50:300, :] = 255
    cv.imshow("fill_binary",image)
    mask = np.ones([402, 402, 1], np.uint8)
    mask[51:301, 51:301] = 0
    # FLOODFILL_MASK_ONLY，mask的指定的位置为零时才填充，不为零不填充
    cv.floodFill(image, mask, (125,125), (0, 0, 255), cv.FLOODFILL_MASK_ONLY)
    cv.imshow("fill", image)


print("----------hello----------")
src = cv.imread("test.jpg")
# cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
# cv.imshow("input image",src)
# fill_color_demo(src)
fill_color_binary()
cv.waitKey(0)
cv.destroyAllWindows()
