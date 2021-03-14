# -*-coding:utf-8-*-
import numpy as np
import cv2 as cv


# 获取图片信息
def get_image_info(image):
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)
    print_data = np.array(image)
    print(print_data)


# 获取视频
def video_demo():
    capture = cv.VideoCapture(0)
    while (True):
        ret, frame = capture.read()
        frame = cv.flip(frame,1)
        cv.imshow("video", frame)
        c = cv.waitKey(50)
        print(c)
        if c == 27:
            cv.destroyAllWindows()
            break


src = cv.imread("timg.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
"""保存图片 第一个参数存储的路径与文件名"""
gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
# cv.imwrite("D:/result.png",src)
cv.imwrite("D:/result.png",gray)
cv.waitKey(0)
cv.destroyAllWindows()

# get_image_info(src)
#
# video_demo()
