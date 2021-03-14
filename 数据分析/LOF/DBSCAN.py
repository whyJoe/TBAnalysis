# -*-coding:utf-8-*-
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics


# 读取csv文件
# def read_csv_data(filename='satellite_image.csv'):
#     data = pd.read_csv(filename)
#     columns = data.columns[:-1]
#     pca_data = func_pca(data[columns])
#     return pca_data


# 使用PCA算法将数据降维
def func_pca(data):
    estimator = PCA(n_components=2)
    data_pca = estimator.fit_transform(data)
    return pd.DataFrame(data_pca)


if __name__ == '__main__':
    X = pd.read_csv('Mulcross.csv')
    data = func_pca(X)
    y_pred = DBSCAN(eps=0.2,min_samples=20).fit_predict(data)

    plt.subplot(221)
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=X.iloc[:,4])
    plt.subplot(222)
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=y_pred)
    plt.show()

    # plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred)
    # a = pd.DataFrame(X.iloc[:, 2])
    # b = pd.DataFrame( y_pred)
    # c = pd.concat([a, b], axis=1)
    # auc_score = metrics.roc_auc_score(c.iloc[:, 0], c.iloc[:, 1])
    # print(auc_score)
    # plt.show()
