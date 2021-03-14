# -*-coding:utf-8-*-
"""
Created on 2019/9/12 11:13
@author: joe
"""
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('Mulcross.csv')
data1 = data.iloc[:, :4]
score = data.iloc[:, 4]
x_train, x_test, y_train, y_test = train_test_split(data1, score, test_size=0.3, random_state=0)

mlp = MLPClassifier(solver='sgd', activation='logistic', alpha=1e-4, hidden_layer_sizes=(10, 10), random_state=1,
                    learning_rate_init=.1,verbose=10)
mlp.fit(x_train,y_train)
y_predict = mlp.predict(x_test)
count = 0
y_test_len = len(y_test)
for i in range(0, len(y_test)):
    if y_predict[i] == y_test.iloc[i]:
        count = count + 1
print('count : ', count, 'y_test_len : ', y_test_len)
print('mpl score:', mlp.score(x_test, y_test))
