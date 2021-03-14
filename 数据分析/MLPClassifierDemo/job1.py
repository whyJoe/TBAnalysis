# -*-coding:utf-8-*-
"""
Created on 2019/9/19 14:32
@author: joe
"""
import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np

# 1.读入数据
data = pd.read_csv('x_train_true.csv', header=None)
test_data = pd.read_csv('x_test_true.csv', header=None)
new_data = pd.read_csv('new_true.csv', header=None, encoding="unicode_escape")
# 输出数据集中正负例的数量
print('训练集样本总量 : ', len(data), '测试集正负例的数量', len(test_data), '新预测集正负例的数量', len(new_data))
print('训练集正负例的数量 : ', data.iloc[:, -1].value_counts())
print('测试集正负例的数量 : ', test_data.iloc[:, -1].value_counts())
print('新预测集正负例的数量 : ', new_data.iloc[:, -1].value_counts())
#########################################


# 2.将数据打乱
data = data.sample(frac=1.0).reset_index(drop=True)
test_data = test_data.sample(frac=1.0).reset_index(drop=True)
new_data = new_data.sample(frac=1.0).reset_index(drop=True)
#########################################


# 3.数据分类
# 训练数据的生成
x = data.iloc[:, :-2]
y = data.iloc[:, -1]
# 测试数据的生成
x_test = test_data.iloc[:, :-2]
y_test = test_data.iloc[:, -1]
# # 新预测数据的生成
x_new = new_data.iloc[:, :-2]
y_new = new_data.iloc[:, -1]
print('---------------------------------------')

mlp = MLPClassifier(solver='adam', activation='relu', alpha=1e-4, hidden_layer_sizes=(60,60,60,60),
                    random_state=1,batch_size=30,
                    learning_rate_init=0.1)

mlp.fit(x,y)
y_predict = mlp.predict(x_test)
tp, fp, tn, fn = [0, 0, 0, 0]
for i in range(0, len(y_test)):
    if y_predict[i] == 1 and y_test[i] == 1:
        tp = tp + 1
    elif y_predict[i] == 1 and y_test[i] == -1:
        fp = fp + 1
    elif y_predict[i] == -1 and y_test[i] == -1:
        tn = tn + 1
    else:
        fn += 1
print('tp : ', tp, 'fp : ', fp, 'tn : ', tn, 'fn : ', fn)
tpr = (tp + tn) / (tp + tn + fp + fn)
print('准确率 : ', tpr)
print('---------------------------------------')

