# -*-coding:utf-8-*-
"""
Created on 2019/9/20 17:40
@author: joe
"""
# https://pypi.doubanio.com/simple/
import tensorflow as tf
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from sklearn.metrics import recall_score
import keras
import numpy as np

np.random.seed(1332)  # for reproducibility

print('Tensorflow Version: {}'.format(tf.__version__))

import pandas as pd
import matplotlib.pyplot as plt

# 1.读入数据
data = pd.read_csv('all_train.csv', header=None)
test_data = pd.read_csv('all_test.csv', header=None)
# 用均值补全数据
data.loc[:, :] = data.loc[:, :].fillna(data.loc[:, :].median())
test_data.loc[:, :] = test_data.loc[:, :].fillna(test_data.loc[:, :].median())
# new_data = pd.read_csv('new_true.csv', header=None)
count = 0
# 输出数据集中正负例的数量
# print('训练集样本总量 : ', len(data), '测试集正负例的数量', len(test_data), '新预测集正负例的数量', len(new_data))
print('训练集样本总量 : ', len(data), '测试集正负例的数量', len(test_data))
print('训练集正负例的数量 : ', data.iloc[:, -1].value_counts())
print('测试集正负例的数量 : ', test_data.iloc[:, -1].value_counts())
# print('新预测集正负例的数量 : ', new_data.iloc[:, -1].value_counts())
#########################################

# 2.将数据打乱
data = data.sample(frac=1.0).sample(frac=1.0).reset_index(drop=True)
test_data = test_data.sample(frac=1.0).reset_index(drop=True)


# new_data = new_data.sample(frac=1.0).reset_index(drop=True)


#########################################

def count_bili(a):
    count = 0
    for i in a:
        if i == 0:
            count += 1
    return count / len(a)


# 3.数据分类
# 训练数据的生成
x = data.iloc[:, :-1]
y = data.iloc[:, -1].replace(-1, 0)
print('分类后训练数据异常比例为 : ', count_bili(y))
# 测试数据的生成
x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1].replace(-1, 0)
print('分类后测试数据异常比例为 : ', count_bili(y_test))

# #########################################
# 过拟合: 在训练数据上得分很高， 在测试数据上得分较低
# 欠拟合: 在训练数据上得分比较低， 在测试数据上得分相对比较低


# 4.网络搭建与参数设置
# ---------------训练模型---------------------
model = keras.Sequential()
model.add(keras.layers.Dense(50, input_shape=(56,), activation='relu'))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# 编译模型
model.compile(optimizer=keras.optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['acc'])
# print(model.summary())
history = model.fit(x, y, epochs=20, batch_size=60, validation_split=0.1, verbose=1)

#########################################################

# 5.评估模型
score = model.evaluate(x_test, y_test)
print("Test score:", score[0])
print('Test accuracy:', score[1])
# 打印样本所属标签
y_pre = model.predict_classes(x_test)
print(y_pre)
del model

tp, fp, tn, fn = [0, 0, 0, 0]
for i in range(0, len(y_pre)):
    if y_pre[i] == 1 and y_test[i] == 1:
        tp = tp + 1
    elif y_pre[i] == 1 and y_test[i] == 0:
        fp = fp + 1
    elif y_pre[i] == 0 and y_test[i] == 0:
        tn = tn + 1
    else:
        fn += 1
try:
    print('tp : ', tp, 'fp : ', fp, 'tn : ', tn, 'fn : ', fn)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    print('tpr : ', tpr, 'fpr : ', fpr)
    print('ACC : ', (tp + tn) / (tp + tn + fp + fn))
    print('查全率 : ', recall_score(y_true=y_test, y_pred=y_pre))
    # print('异常点准确率 : ', tn / (tn + fn), '查全率 : ', tn / (tn + fp))
except BaseException:
    pass
# 6.保存模型
# -----save前------
# print('before save model:', model.predict(x_test[0:2]))
# model.save('model.hdf5')
# del model

# -----save后------
# with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
#     model = load_model('model.hdf5')
# # print('after save:', model.predict(x_test[0:2]))
# score = model.evaluate(x_test, y_test)
# print("Test score:", score[0])
# print('Test accuracy:', score[1])
# # ---------------------------------------


from sklearn.metrics import roc_curve, auc

# y_pred = model.predict(x_new).ravel()
# acc = metrics.accuracy_score(y_test, y_pre)
# print(acc)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, 'b', label='Keras AUC = %0.2f' % roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.show()
# plt.clf()

# 7.绘图 acc和loss图，观察有没有过拟合
# plt.title('acc')
# plt.plot(history.epoch, history.history.get('acc'), label='acc')
# plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
# plt.legend()
# plt.show()
#
# plt.clf()
# plt.plot(history.epoch, history.history.get('loss'), label='loss')
# plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
# plt.legend()
# plt.show()

# with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
#     model = load_model('model.hdf5')
# y_pre = model.predict(x_test,shuffle=False)

# from sklearn.decomposition import PCA
#
# plt.clf()
# X_test = test_data.iloc[:, :-2]
# y_test = test_data.iloc[:, -1]
# estimator = PCA(n_components=2)
# plt_test = estimator.fit_transform(x_test)
# plt.subplot(221)
# plt.title('1')
# arr = []
# for i in range(len(y_test)):
#     if y_test[i] == -1:
#         arr.append('red')
#     else:
#         arr.append('blue')
# plt.scatter(plt_test[:, 0], plt_test[:, 1], c=arr)
# plt.subplot(222)
# plt.title('2')
# arr1 = []
# for i in y_pre:
#     if i == 0:
#         arr1.append('red')
#     else:
#         arr1.append('blue')
# plt.scatter(plt_test[:, 0], plt_test[:, 1], c=arr1)
# plt.show()
