# # -*-coding:utf-8-*-
# from sklearn.datasets import load_iris
# import pandas as pd
#
# data = load_iris()
# print(dir(data))
# print(data.feature_names)
# print(data.target_names)
# a = pd.DataFrame(data=data.data,columns=data.feature_names)
# print(a)
#
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
#
#
# X = data.data  # 只包括样本的特征，150x4
# y = data.target  # 样本的类型，[0, 1, 2]
# features = data.feature_names  # 4个特征的名称
# targets = data.target_names  # 3类鸢尾花的名称，跟y中的3个数字对应
#
# plt.figure(figsize=(10, 4))
# plt.plot(X[:, 2][y==0], X[:, 3][y==0], 'bs', label=targets[0])
# plt.plot(X[:, 2][y==1], X[:, 3][y==1], 'kx', label=targets[1])
# plt.plot(X[:, 2][y==2], X[:, 3][y==2], 'ro', label=targets[2])
# plt.xlabel(features[2])
# plt.ylabel(features[3])
# plt.title('Iris Data Set')
# plt.legend()
# plt.savefig('Iris Data Set.png', dpi=200)
# plt.show()
#
# iris = load_iris()
# print (iris.data)          #输出数据集
# print (iris.target)         #输出真实标签
# #获取花卉两列数据集
# DD = iris.data
# X = [x[0] for x in DD]
# print(X)
# Y = [x[1] for x in DD]
# print(Y)
