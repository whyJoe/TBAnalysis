# -*-coding:utf-8-*-
"""
Created on 2019/8/21 17:47
@author: joe
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

x, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                           n_redundant=2, n_repeated=0, n_classes=2,
                           n_clusters_per_class=2, weights=None,
                           flip_y=0.01, class_sep=1.0, hypercube=True,
                           shuffle=True, random_state=10)
plt.scatter(x[:, 0], x[:, 1], c=y, s=7)  # 共20个特征维度，此处仅使用两个维度作图演示
plt.savefig('make_classification.png')
plt.show()

count = 0
num = 0
for i in y:
    if i == 0:
        count += 1
    else:
        num += 1
print(count)
print(num)
