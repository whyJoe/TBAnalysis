# -*-coding:utf-8-*-
"""
Created on 2019/9/20 17:40
@author: joe
"""
import tensorflow as tf
import keras
import numpy as np
from matplotlib.pyplot import plot as plt

np.random.seed(1337)  # for reproducibility

print('Tensorflow Version: {}'.format(tf.__version__))

from tf_demo.test import UseData

x,x_test,y,y_test = UseData.data()
# #########################################
# 过拟合: 在训练数据上得分很高， 在测试数据上得分较低
# 欠拟合: 在训练数据上得分比较低， 在测试数据上得分相对比较低

# 4.网络搭建与参数设置
# ---------------训练模型---------------------
model = keras.Sequential()
model.add(keras.layers.Dense(500, input_shape=(54,), activation='relu'))
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.Dense(500, activation='relu'))
# model.add(keras.layers.Dense(10, input_shape=(58,), activation='relu'))
# model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# 编译模型
model.compile(optimizer=keras.optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(x,y, epochs=1000, batch_size=60, validation_split=0.1, verbose=1)

#########################################################

# 5.评估模型
# score = model.evaluate(x_test, y_test)
# print("Test score:", score[0])
# print('Test accuracy:', score[1])
y_pre = model.predict(x_test)
del model
aa = []
count = 0
for i in y_pre:
    if i[0] > 0.5:
        count += 1
        aa.append(1)
    else:
        aa.append(0)
tp, fp, tn, fn = [0, 0, 0, 0]
for i in range(0, len(y_pre)):
    if aa[i] == 1 and y_test[i] == 1:
        tp = tp + 1
    elif aa[i] == 1 and y_test[i] == 0:
        fp = fp + 1
    elif aa[i] == 0 and y_test[i] == 0:
        tn = tn + 1
    else:
        fn += 1
print('tp : ', tp, 'fp : ', fp, 'tn : ', tn, 'fn : ', fn)
tpr = (tp + tn) / (tp + tn + fp + fn)

print('异常点准确率 : ',tn/(tn+fn),'查全率 : ',tn/(tn+fp),)
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


# y_pred = model.predict(x_new).ravel()
# fpr, tpr, threshold = roc_curve(y_new, y_pred)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, 'b', label='Keras AUC = %0.2f' % roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.show()
# plt.clf()

# 7.绘图 acc和loss图，观察有没有过拟合
plt.title('acc')
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()
plt.show()

plt.clf()
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()

# with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
#     model = load_model('model.hdf5')
# y_pre = model.predict(x_test,shuffle=False)
