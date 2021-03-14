# -*-coding:utf-8-*-
"""
Created on 2019/9/16 14:23
@author: joe
"""
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score


# 将原始数据转化为二维数据画图
# 数据丢失太多，舍弃
def plt_data(data):
    X_test = data1[data1['11'] == 2].iloc[:, 1:-2]
    y_test = data1[data1['11'] == 2].iloc[:, -1]
    estimator = PCA(n_components=2)
    score = data.iloc[:, -1]
    plt_test = estimator.fit_transform(X_test)
    plt.subplot(221)
    plt.title('original_test_picture')
    plt.scatter(plt_test[:, 0], plt_test[:, 1], c=y_test)
    result = estimator.fit_transform(data.iloc[:, 1:-2])
    plt.subplot(222)
    plt.title('original picture')
    plt.scatter(result[:, 0], result[:, 1], c=score, linewidths=0.001)


def iforest_predict():
    X_train = data1[data1.iloc[:, -2] == 1].iloc[:, :-2]
    X_test = data1[data1.iloc[:, -2] == 2].iloc[:, :-2]
    y_train = data1[data1.iloc[:, -2] == 1].iloc[:, -1]
    y_test = data1[data1.iloc[:, -2] == 2].iloc[:, -1]

    ifor = IsolationForest(behaviour='new', max_samples=276,
                           random_state=0, contamination=0.04, n_estimators=10)
    ifor.fit(X_train)
    y_predict = ifor.predict(X_test)
    tp, fp, tn, fn = [0, 0, 0, 0]
    for i in range(0, len(y_test)):
        if y_predict[i] == 1 and y_test.iloc[i] == 1:
            tp += 1
        elif y_predict[i] == 1 and y_test.iloc[i] == -1:
            fp += 1
        elif y_predict[i] == -1 and y_test.iloc[i] == 1:
            tn += 1
        else:
            fn += 1

    print('tp : ', tp, 'fp : ', fp, 'tn : ', tn, 'fn : ', fn)
    print('准确率 : ', (tp + tn) / (tp + tn + fp + fn))
    print('错误率 : ', (fp + fn) / (tp + tn + fp + fn))
    print('特效度 : ', tn / (fp + tn))
    print('roc:', roc_auc_score(y_test, y_predict))
    df = pd.DataFrame(y_predict, y_test)
    print(df)
    # estimator = PCA(n_components=2)
    # plt_test = estimator.fit_transform(X_test)
    # plt.subplot(223)
    # plt.scatter(plt_test[:, 0], plt_test[:, 1], c=y_predict, linewidths=0.001)
    # plt.show()


def mlp_test():
    X_train = data1[data1.iloc[:, -2] == 1].iloc[:, 1:-2]
    X_test = data1[data1.iloc[:, -2] == 2].iloc[:, 1:-2]
    y_train = data1[data1.iloc[:, -2] == 1].iloc[:, -1]
    y_test = data1[data1.iloc[:, -2] == 2].iloc[:, -1]

    mlp = MLPClassifier(solver='adam', activation='identity', alpha=1e-4, hidden_layer_sizes=(200, 200),
                        random_state=1,
                        learning_rate_init=.1)
    mlp.fit(X_train, y_train)
    y_predict = mlp.predict(X_test)
    count = 0
    for i in y_predict:
        if i == -1:
            count += 1
    df = pd.DataFrame(y_predict, y_test)
    print(df)
    tp, fp, tn, fn = [0, 0, 0, 0]
    for i in range(0, len(y_test)):
        if y_predict[i] == 1 and y_test.iloc[i] == 1:
            tp = tp + 1
        elif y_predict[i] == 1 and y_test.iloc[i] == -1:
            fp = fp + 1
        elif y_predict[i] == -1 and y_test.iloc[i] == -1:
            tn = tn + 1
        else:
            fn += 1
    print('mlp score : ', mlp.score(X_test, y_test))
    print('tp : ', tp, 'fp : ', fp, 'tn : ', tn, 'fn : ', fn)
    print('真正率tpr : ', tp / (tp + fp))
    # 真负率，可理解为错误的被判断为错误的
    print('真负率 : ', tn / (tn + fp))


if __name__ == '__main__':
    # 读取csv的数据,后面必须添加encoding="unicode_escape"
    data1 = pd.read_csv('fivetest.csv', low_memory=False, encoding="unicode_escape")
    columns = [str(i) for i in range(data1.shape[1])]
    data1.columns = columns
    data1.fillna(0)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 测试iforest
    plt_data(data1)
    # iforest_predict()
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 测试神经网络
    # mlp_test()
