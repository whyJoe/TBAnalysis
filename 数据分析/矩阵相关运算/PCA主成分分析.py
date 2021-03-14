import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# 1.初始化一个线性矩阵并求秩
# M = np.array([[1, 2], [2, 4]])
# np.linalg.matrix_rank(M, tol=None)

# 2.读取训练数据与测试数据集
digits_train = pd.read_csv('optdigits.tra',
                           header=None)
# digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',
#                           header=None)
print("训练数据集", digits_train.shape)
print(digits_train)
# print("测试数据集", digits_test.shape)

# 3.将数据降到2维并可视化

# 3.1 分割训练数据的特征向量和标记
# X_digits = digits_train[np.arange(64)]
# y_digits = digits_train[64]
# print('y_digits',type(y_digits))
#
# 3.2 PCA降维——>二维
# estimator = PCA(n_components=2)
# X_pca = estimator.fit_transform(X_digits)
#
#
# # 3.3 显示这10类手写体数字图片经PCA压缩后的2为空间分布
# def plot_pca_scatter():
#     colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
#     for i in range(len(colors)):
#         px = X_pca[:, 0][y_digits.values == i]
#         py = X_pca[:, 1][y_digits.values == i]
#         plt.scatter(px, py, c=colors[i])
#     plt.legend(np.arange(0, 10).astype(str))
#     plt.xlabel('First Principal Component')
#     plt.ylabel('Second Principal Component')
#     plt.show()
#
#
# plot_pca_scatter()
#
# # 4.用SVM分别对原始空间的数据(64维)和降到20维的数据进行训练，预测
#
# # 4.1 对训练数据/测试数据进行特征向量与分类标签的分离
# X_train = digits_train[np.arange(64)]
# y_train = digits_train[64]
# X_test = digits_test[np.arange(64)]
# y_test = digits_test[64]
#
# # 4.2 用SVM对64维数据进行训练
# svc = LinearSVC()  # 初始化线性核的支持向量机的分类器
# svc.fit(X_train, y_train)
# y_pred = svc.predict(X_test)
#
# # 4.3 用SVM对20维数据进行训练
# estimator = PCA(n_components=20)
# pca_X_train = estimator.fit_transform(X_train)  # 利用训练特征决定20个正交维度的方向，并转化原训练特征
# pca_X_test = estimator.transform(X_test)
#
# psc_svc = LinearSVC()
# psc_svc.fit(pca_X_train,y_train)
# pca_y_pred = psc_svc.predict(pca_X_test)
#
# # 获取结果报告
# # 输出用64维度训练的结果
# print(svc.score(X_test,y_test))
# print(classification_report(y_test,y_pred,target_names=np.arange(10).astype(str)))
#
# # 输出用20维度训练的结果
# print(psc_svc.score(pca_X_test,y_test))
# print(classification_report(y_test,pca_y_pred,target_names=np.arange(10).astype(str)))
