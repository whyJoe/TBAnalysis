# -*-coding:utf-8-*-
"""
Created on 2019/8/20 14:53
@author: joe
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import csv
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

rng = np.random.RandomState(50)


def iforest_predict_pca(max_samples):
    data1 = pd.read_csv("db2.csv")
    data1 = data1.dropna(axis=1)
    estimator = PCA(n_components=2)
    data = data1.iloc[:, :5]
    score = data1.iloc[:, 5]
    data1 = estimator.fit_transform(data)
    x_train, x_test, y_train, y_test = train_test_split(data1, score, test_size=0.3,
                                                        random_state=0)
    ifor = IsolationForest(behaviour='new', max_samples=max_samples,
                           random_state=rng, contamination="auto")
    ifor.fit(x_train)
    pred_test = ifor.predict(x_test)
    # pred_test = ifor.fit_predict(x_test)
    # score = ifor.decision_function(x_test)
    print('roc = ', roc_auc_score(y_test, pred_test))
    # count = 1
    # for i in score:
    #     count += 1
    #     print(count, "    ", i)
    return roc_auc_score(y_test, pred_test)


def iforest_predict(max_samples,n):
    data1 = pd.read_csv("Mulcross.csv")
    data1 = data1.dropna(axis=1)
    estimator = PCA(n_components=2)
    data = data1.iloc[:, :4]
    score = data1.iloc[:, 4]
    data1 = estimator.fit_transform(data)
    ifor = IsolationForest(behaviour='new', max_samples=max_samples,
                           random_state=rng, contamination=0.1,n_estimators=n)
    ifor.fit(data)
    pred_test = ifor.predict(data)
    print('roc = ', roc_auc_score(score, pred_test))




    # 画对比图
    # plt.subplot(221)
    # plt.scatter(data1[:, 0], data1[:, 1], c=score)
    # plt.subplot(222)
    # plt.scatter(data1[:, 0], data1[:, 1], c=pred_test)
    # plt.show()
    return roc_auc_score(score, pred_test)


def iforest_predict_no(max_samples,n):
    data1 = pd.read_csv("haberman.csv")
    data1 = data1.dropna(axis=1)
    data = data1.iloc[:, :3]
    score = data1.iloc[:, 3]
    estimator = PCA(n_components=2)
    data1 = estimator.fit_transform(data)

    ifor = IsolationForest(behaviour='new', max_samples=max_samples,
                           random_state=rng, contamination="auto",n_estimators=n)

    pred_test = ifor.fit_predict(data)
    print('auc = ', roc_auc_score(score, pred_test))

    plt.subplot(221)
    plt.scatter(data1[:, 0], data1[:, 1], c=score)
    plt.subplot(222)
    plt.scatter(data1[:, 0], data1[:, 1], c=pred_test[:])
    plt.show()
    return roc_auc_score(score, pred_test)


# iforest_predict(20,500)

out = open('if_have_n100.csv', 'a', newline='')
csv_write = csv.writer(out, dialect='excel')
for i in range(10, 100,10):
        start = time.clock()
        roc_val = iforest_predict(i,100)
        end = time.clock()
        t = end - start
        stu = ['max_samples= ',i,roc_val, t]
        csv_write.writerow(stu)
#
# print('===================================')
# out = open('db2_if_no.csv', 'wb', newline='')
# csv_write = csv.writer(out, dialect='excel')
# for i in range(126, 300):
#     start = time.clock()
#     roc_val = iforest_predict_no(i)
#     end = time.clock()
#     t = end - start
#     stu = [i, roc_val, t]
#     csv_write.writerow(stu)

# ps = pd.read_csv('db2_if_have.csv')
# first_twelve = ps.iloc[:,:2]
# print(len(first_twelve))
# # sum = 0
# # for i in first_twelve:
# #     sum = sum+i
# # print(sum/len(first_twelve))
# plt.plot(first_twelve.iloc[:,0],first_twelve.iloc[:,1])
# plt.xlabel('The size of K')
# plt.ylabel('The size of AUC')
# plt.show()
