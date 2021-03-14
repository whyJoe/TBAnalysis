# -*-coding:utf-8-*-
"""
Created on 2019/9/12 10:55
@author: joe
"""
from sklearn.neural_network import MLPClassifier

X = [[0., 0.], [1., 1.],[0,1]]
y = [0, 1,1]
mlp = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,5),random_state=1)
mlp.fit(X, y)
result = mlp.predict(X)
print(result)
print(mlp.n_layers_)
print(mlp.n_iter_)
print(mlp.loss_)
print(mlp.out_activation_)
