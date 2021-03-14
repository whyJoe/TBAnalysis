# -*-coding:utf-8-*-
"""
Created on 2019/10/22 15:56
@author: joe
"""
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA


def data():
    # 默认均值填补
    imp_mean = SimpleImputer()
    # 使用均值填补
    imp_median = SimpleImputer(strategy="median")

    # 1.读入数据
    data = pd.read_csv(r'train73.csv', header=None)
    test_data = pd.read_csv(r'test73.csv', header=None)
    # data = pd.read_csv(r'all_train.csv', header=None)
    # test_data = pd.read_csv(r'all_test.csv', header=None)
    # data = imp_mean.fit_transform(data)
    data = data.sample(frac=1.0).reset_index(drop=True)
    print(data)
    # 查看数据的基本情况
    # 用均值填补
    data.loc[:, :] = data.loc[:, :].fillna(data.loc[:, :].median())
    test_data.loc[:, :] = test_data.loc[:, :].fillna(test_data.loc[:, :].median())
    xtrain = data.iloc[:, :-2]
    xtrain[xtrain < 0] = 0
    ytrain = data.iloc[:, -2].replace(-1, 0)
    xtest = test_data.iloc[:, :-2]
    xtest[xtest < 0] = 0
    ytest = test_data.iloc[:, -2].replace(-1, 0)
    print('训练集正负例的数量 : ', data.iloc[:, -2].value_counts())
    print('测试集正负例的数量 : ', test_data.iloc[:, -2].value_counts())
    # xtrain = SelectKBest(chi2, k=50).fit_transform(xtrain, ytrain)
    # xtest = SelectKBest(chi2, k=50).fit_transform(xtest, ytest)
    # xtrain = StandardScaler().fit_transform(xtrain)
    # xtest = StandardScaler().fit_transform(xtest)
    xtrain = Normalizer().fit_transform(xtrain)
    # estimator = PCA(n_components=3)
    # xtrain = estimator.fit_transform(xtrain)
    xtest = Normalizer().fit_transform(xtest)
    # xtest = estimator.fit_transform(xtest)
    return xtrain, xtest, ytrain, ytest
