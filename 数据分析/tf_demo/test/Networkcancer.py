# -*-coding:utf-8-*-
"""
Created on 2019/11/7 14:03
@author: joe
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt

cancer = load_breast_cancer()
print(cancer.target)
n = 0
for i in cancer.target:
    if i == 0:
        n = n + 1
print(n)
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(5, 5))
mlp.fit(x_train, y_train)
print("neural network:")
print("accuracy on the training subset:{:.3f}".format(mlp.score(x_train, y_train)))
print("accuracy on the test subset:{:.3f}".format(mlp.score(x_test, y_test)))
print("标准化前", "=" * 30)
scaler = StandardScaler()
x_train_scaled = scaler.fit(x_train).transform(x_train)
x_test_scaled = scaler.fit(x_test).transform(x_test)

mlp_scaled = MLPClassifier(max_iter=1000, random_state=42)
mlp_scaled.fit(x_train_scaled, y_train)
print("neural network after scaled:")
print("accuracy on the training subset:{:.3f}".format(mlp_scaled.score(x_train_scaled, y_train)))
print("accuracy on the test subset:{:.3f}".format(mlp_scaled.score(x_test_scaled, y_test)))
print("标准化后", "=" * 30)
mlp_scaled2 = MLPClassifier(max_iter=1000, alpha=1, random_state=42)
mlp_scaled.fit(x_train_scaled, y_train)
print("neural network after scaled and alpha change to 1:")
print("accuracy on the training subset:{:.3f}".format(mlp_scaled.score(x_train_scaled, y_train)))
print("accuracy on the test subset:{:.3f}".format(mlp_scaled.score(x_test_scaled, y_test)))

