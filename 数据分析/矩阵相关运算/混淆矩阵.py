# -*-coding: utf-8 -*-
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


# """
# 	Python格式化输出混淆矩阵
# 	:param confusion_matrix: 混淆矩阵，一个numpy矩阵，元素均为整型
# 	:param type_name: 类别名称，一个字符串列表，默认为None
# 	:param placeholder_length: 占位符宽度，即每个数字占几位，用于对齐，默认为5
# """
#
#
def format_print_confusion_matrix(confusion_matrix, type_name=None, placeholder_length=5):
    if type_name != None:
        type_name.insert(0, 'T \ P')  # 头部插入一个元素补齐
        for tn in type_name:
            fm = '%' + str(placeholder_length) + 's'
            print(fm % tn, end='')  # 不换行输出每一列表头
        print('\n')

    for i, cm in enumerate(confusion_matrix):
        if type_name != None:
            fm = '%' + str(placeholder_length) + 's'
            print(fm % type_name[i + 1], end='')  # 不换行输出每一行表头

        for c in cm:
            fm = '%' + str(placeholder_length) + 'd'
            print(fm % c, end='')  # 不换行输出每一行元素
        print('\n')


if __name__ == '__main__':
    confusion_matrix_example = np.array([[3, 1],
                                         [3, 3]])
    type_name_example = ['狗', '猫']
    format_print_confusion_matrix(confusion_matrix_example, type_name_example, 7)

from sklearn import datasets  # 引入数据集

# 构造的各种参数可以根据自己需要调整
X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=1)

###绘制构造的数据###
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(X, y)
plt.show()
