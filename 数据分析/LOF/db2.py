# -*-coding:utf-8-*-
import numpy as np
import time
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn import metrics
from sklearn import model_selection
import csv
import time
import seaborn as sns


# 使用PCA算法将数据降维
def func_pca(data):
    estimator = PCA(n_components=2)
    data_pca = estimator.fit_transform(data)
    return pd.DataFrame(data_pca)


# 打印数组的形状
def func_printshape(arr):
    print(arr.shape)


def grid_lof(filename, w, k):
    grid_data = read_csv_data(filename=filename, w=w)
    ready_lof = np.array([0, 0, 0, 0])
    for i in grid_data:
        if len(i):
            for j in i:
                ready_lof = np.vstack((j, ready_lof))
        else:
            continue
    #  异常点比例
    con = 0
    for i in range(len(ready_lof[:, 2])):
        if ready_lof[i, 2] == -1:
            con += 1
    con_pre = con / len(ready_lof[:, 2])
    model = LocalOutlierFactor(n_neighbors=k, contamination=con_pre, )
    x_train, x_test, y_train, y_test = train_test_split(ready_lof[:, :2], ready_lof[:, 2], test_size=0.01,
                                                        random_state=0)
    model.fit(x_train)

    # 局部离群因子计算
    lofyinzi = model.negative_outlier_factor_
    pd_lof = pd.DataFrame(lofyinzi, columns=['column1'])
    pd_lof['index'] = np.arange(len(pd_lof))
    pd_lof.sort_values('column1', inplace=True)

    y_score = model.fit_predict(ready_lof[:, :2])
    tp, fp, tn, fn = [0, 0, 0, 0]
    for i in range(len(y_score)):
        if ready_lof[i, 2] == 1:
            if y_score[i] == 1:
                tp = tp + 1
            else:
                tn = tn + 1
        elif ready_lof[i, 2] == -1:
            if y_score[i] == -1:
                fn = fn + 1
            else:
                fp = fp + 1
    print('tp : ', tp, 'tn : ', tn, 'fp : ', fp, 'fn : ', fn)
    ready_lof[-1, 2] = -1
    y_test_int = ready_lof[:, 2].astype(int)
    a = pd.DataFrame(y_test_int)
    b = pd.DataFrame(y_score)
    c = pd.concat([a, b], axis=1)
    auc_score = metrics.roc_auc_score(c.iloc[:, 0], c.iloc[:, 1])
    print('grid_lof k : ', k, ' w : ', w, ' auc : ', auc_score)
    # print('------------------------------------')


def lof(filename, k):
    print("==================请等待=================")
    # data = read_csv_data_notgrid(filename=filename)
    data = pd.read_csv(filename)
    # ready_lof = np.array([0, 0, 0, 0, 0, 0])
    # for i in range(len(data)):
    #     ready_lof = np.vstack((data.iloc[i, :], ready_lof))
    model = LocalOutlierFactor(n_neighbors=k, contamination=0.32)
    # model.fit(data[:, :4])
    y_score = model.fit_predict(data.iloc[:, :4])
    # tp, fp, tn, fn = [0, 0, 0, 0]
    # for i in range(len(y_score)):
    #     if ready_lof[i, 4] == 1:
    #         if y_score[i] == 1:
    #             tp = tp + 1
    #         else:
    #             tn = tn + 1
    #     elif ready_lof[i, 4] == -1:
    #         if y_score[i] == -1:
    #             fn = fn + 1
    #         else:
    #             fp = fp + 1

    auc_score = metrics.roc_auc_score(data.iloc[:, 4], y_score)
    print('lof k :', k, 'auc : ', auc_score)

    # e = PCA(n_components=2)
    # data2 = e.fit_transform(data.iloc[:, :3])
    # plt.subplot(221)
    # plt.scatter(data2[:, 0], data2[:, 1], c=data.iloc[:, 3])
    # plt.subplot(222)
    # plt.scatter(data2[:, 0], data2[:, 1], c=y_score)
    # plt.show()
    return auc_score


def func_test():
    start = time.clock()
    lof('Mulcross.csv', k=130)
    end = time.clock()
    t = end - start
    print('%f秒' % t)


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


def func_grid(data_pca, data_index, wj):
    plt.figure()
    plt.subplot(221)
    plt.title('Original Pic')
    plt.scatter(y=data_pca[:, 0], x=data_pca[:, 1], c=data_index[:, 0])
    X_min = min(data_pca[:, 0])
    X_max = max(data_pca[:, 0])
    Y_min = min(data_pca[:, 1])
    Y_max = max(data_pca[:, 1])
    # print('------------------------------------')
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
                                                                  - Y_min) // (Y_len / wj))].append(
                np.hstack((data_pca[i][:], data_index[i][:])))
            cout += 1

        except:
            w[int((data_pca[i, 0] - X_min) // (X_len / wj)) - 1][
                int((data_pca[i, 1] - Y_min) // (Y_len / wj)) - 1].append(
                np.hstack((data_pca[i][:], data_index[i][:])))
            cout += 1
    # print('X_min:', X_min, '\nX_max:', X_max, '\nY_min:', Y_min, '\nY_max:', Y_max)
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
                c = int(i[j][2])
                if c == -1:
                    color = '#00CED1'
                    plt.scatter(x=i[j][1], y=i[j][0], c=color, label='异常点')
                else:
                    color = '#DC143C'
                    plt.scatter(x=i[j][1], y=i[j][0], c=color, label='正常点')
                count += 1
        else:
            notnull_arr.remove(i)
    # print('notnull_arr=============',notnull_arr)
    print('筛选后数量为:', count)
    plt.show()
    return notnull_arr


# 读取csv文件
# filename:读取文件名(.csv)，w:划分网格数
def read_csv_data(filename, w):
    data = pd.read_csv(filename)
    print("从csv中读取完数据。。。")
    data_columns = data.columns[:-1]
    target = data.columns[-1]
    pca_dataFrame = func_pca(data[data_columns])
    target_pd = pd.DataFrame(data[target])
    index_pd = pd.DataFrame(np.arange(len(data)), columns=['index'])
    data_concat = pd.concat([pca_dataFrame, target_pd, index_pd], axis=1)
    a = data_concat.iloc[:, :2]
    b = data_concat.iloc[:, 2:]
    grid_data = func_grid(a.values, b.values, wj=w)
    print("网格划分完毕。。。")
    return grid_data


# 不用网格划分
def read_csv_data_notgrid(filename):
    data = pd.read_csv(filename)
    data_columns = data.columns[:-1]
    target = data.columns[-1]
    # 降维
    # pca_dataFrame = func_pca(data[data_columns])
    pca_dataFrame = data[data_columns]
    target_pd = pd.DataFrame(data[target])
    index_pd = pd.DataFrame(np.arange(len(pca_dataFrame)), columns=['index'])
    data_concat = pd.concat([pca_dataFrame.iloc[:, :4], target_pd, index_pd], axis=1)
    a = data_concat.iloc[:, :4]
    b = data_concat.iloc[:, 4:]
    # plt.figure()
    # plt.scatter(y=pca_dataFrame.iloc[:, 0], x=pca_dataFrame.iloc[:, 1])
    # plt.show()
    return data_concat


if __name__ == '__main__':
    # out = open('M_lof.csv', 'a',newline='')
    # csv_write = csv.writer(out, dialect='excel')
    # for i in range(50,401,50):
    #     start = time.clock()
    #     roc_val = lof('Mulcross.csv', i)
    #     end = time.clock()
    #     t = end - start
    #     stu = [i, roc_val, t]
    #     csv_write.writerow(stu)

    data = pd.read_csv('if_have_n100.csv')
    # k = data.iloc[:, ]
    auc = data.iloc[:, 2]
    time = data.iloc[:, 3]
    max_samples = data.iloc[:, 1]
    df = pd.DataFrame({'max_samples': max_samples, 'auc': auc})
    sns.barplot(x='max_samples', y='auc', data=df)
    plt.show()

    # ps = pd.read_csv('haberman_lof.csv')
    # first_twelve = ps.iloc[:, :2]
    # plt.plot(first_twelve.iloc[:, 0], first_twelve.iloc[:, 1])
    # plt.xlabel('The size of K')
    # plt.ylabel('The size of AUC')
    # plt.show()
