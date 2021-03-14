# -*-coding:utf-8-*-
import numpy as np
import time
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn import model_selection


# 使用PCA算法将数据降维
def func_pca(data, n):
    estimator = PCA(n_components=2)
    data_pca = estimator.fit_transform(data)
    return data_pca


# 打印数组的形状
def func_printshape(arr):
    print(arr.shape)


# 使用lof算法并进行可视化
def func_Lof(data, n_neighbors=5, contamination=0.1, novelty=True):
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination,
                               novelty=novelty)  # 定义一个LOF模型，异常比例是10%
    model.fit(data)
    y = model.predict(data)
    plt.subplot(223)
    plt.title('GridLof Pic')
    # 若样本点正常，返回1，不正常，返回-1
    plt.scatter(data[:, 0], data[:, 1], c=y)  # 样本点的颜色由y值决定
    # plt.grid(True)
    # plt.grid(color='r', linewidth='2', linestyle='--', axis='both')  # 画出网格
    plt.show()


def func_test(data, n_neighbors, contamination, novelty=True):
    start = time.clock()
    func_Lof(data, n_neighbors=n_neighbors, contamination=contamination, novelty=novelty)
    end = time.clock()
    t = end - start
    print('k=%d : %f秒' % (n_neighbors, end))


def return_tarin_test():
    iris = load_iris()
    X = iris.data
    # func_printshape(iris)
    y = iris.target

    # 变为二分类
    X, y = X[y != 2], y[y != 2]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    # np.c_ :按行连接两个矩阵
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.3, random_state=0)
    return X_train, X_test, y_train, y_test


"""
将数据从网格中划分出来
Param:data:待划分的二维数据， wj:网格划分大小，越小网格越大
"""


def func_grid(data, wj):
    plt.figure(figsize=(200, 200))
    data_pca = func_pca(data, 2)
    plt.subplot(221)
    plt.title('Original Pic')
    plt.scatter(y=data_pca[:, 0], x=data_pca[:, 1])
    X_min = min(data_pca[:, 0])
    X_max = max(data_pca[:, 0])
    Y_min = min(data_pca[:, 1])
    Y_max = max(data_pca[:, 1])
    print('X_min:', X_min, '\nX_max:', X_max, '\nY_min:', Y_min, '\nY_max:', Y_max)
    # X轴上的长度，Y轴上的长度
    X_len = X_max - X_min
    Y_len = Y_max - Y_min
    w = [[[] for i in range(wj)] for j in range(wj)]  # 存放数据的二维数组
    w_size = len(w)

    cout = 0  # 计算输入数据样本总数量
    # 将数据放入格子
    for i in range(len(data_pca)):
        try:
            w[int((data_pca[i, 0] - X_min) // (X_len / wj))][int((data_pca[i, 1]
                                                                  - Y_min) // (Y_len / wj))].append(data_pca[i][:])
            cout += 1
        except:
            w[int((data_pca[i, 0] - X_min) // (X_len / wj)) - 1][
                int((data_pca[i, 1] - Y_min) // (Y_len / wj)) - 1].append(
                data_pca[i, :])
            cout += 1
    print('样本总数', cout)

    notnull_arr = []  # 存放检测出来周围格子都不是空的点
    # 开始将数据筛选出来
    for i in range(w_size):
        for j in range(w_size):
            # 划分四个顶点情况
            if i == 0 and 0 == j:
                if (len(w[i + 1][j + 1]) > 0 and len(w[i][j + 1]) > 0 and len(w[i + 1][j]) > 0) == 0:
                    notnull_arr.append(w[i][j])
            elif i == (w_size - 1) and j == 0:
                if (len(w[i][j + 1]) > 0 and len(w[i - 1][j + 1]) > 0 and len(w[i - 1][j + 1]) > 0) == 0:
                    notnull_arr.append(w[i][j])
            elif i == 0 and j == (w_size - 1):
                if (len(w[i][j - 1]) > 0 and len(w[i + 1][j - 1]) > 0 and len(w[i + 1][j]) > 0) == 0:
                    notnull_arr.append(w[i][j])
            elif i == (w_size - 1) and j == (w_size - 1):
                if (len(w[i][j - 1]) > 0 and len(w[i - 1][j - 1]) > 0 and len(w[i - 1][j]) > 0) == 0:
                    notnull_arr.append(w[i][j])
            # 划分边值情况
            elif i == 0:
                if (len(w[i][j - 1]) > 0 and len(w[i][j + 1]) > 0 and len(w[i + 1][j + 1]) > 0 and len(
                        w[i + 1][j]) > 0 and len(w[i + 1][j - 1]) > 0) == 0:
                    notnull_arr.append(w[i][j])
            elif j == 0:
                if (len(w[i][j + 1]) > 0 and len(w[i - 1][j + 1]) > 0 and len(w[i - 1][j]) > 0 and len(
                        w[i + 1][j + 1]) > 0 and len(w[i + 1][j]) > 0) == 0:
                    notnull_arr.append(w[i][j])
            elif i == (w_size - 1):
                if (len(w[i][j - 1]) > 0 and len(w[i][j + 1]) > 0 and len(w[i - 1][j + 1]) > 0 and len(
                        w[i - 1][j - 1]) > 0 and len(w[i - 1][j]) > 0) == 0:
                    notnull_arr.append(w[i][j])
            elif j == (w_size - 1):
                if (len(w[i - 1][j]) > 0 and len(w[i - 1][j - 1]) > 0 and len(w[i][j - 1]) > 0 and len(
                        w[i + 1][j - 1]) > 0 and len(w[i + 1][j]) > 0) == 0:
                    notnull_arr.append(w[i][j])

            # 正常点
            else:
                if (len(w[i - 1][j - 1]) > 0 and len(w[i - 1][j]) > 0 and len(w[i - 1][j + 1]) > 0 and len(
                        w[i][j - 1]) > 0 and len(
                    w[i][j + 1]) > 0 and len(w[i + 1][j - 1]) > 0 and len(w[i + 1][j]) > 0 and len(w[i + 1][j + 1]) > 0) \
                        == 0:
                    notnull_arr.append(w[i][j])
    # 筛选后的点有多少个
    count = 0
    plt.subplot(222)
    plt.title('Grid Pic')
    for i in notnull_arr:
        if len(i):
            for j in range(len(i)):
                plt.scatter(x=i[j][1], y=i[j][0])
                count += 1
        else:
            notnull_arr.remove(i)
    print('筛选后数量为:', count)
    # plt.show()
    return notnull_arr


# 读取csv文件
def read_csv_data(filename='satellite_image.csv'):
    data = pd.read_csv(filename)
    columns = data.columns[:-1]
    pca_data = func_pca(data[columns], 2)
    return pca_data


if __name__ == '__main__':
    abs(1)