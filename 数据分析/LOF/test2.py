# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:23:32 2019

@author: kenny
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

rng = np.random.RandomState(50)
# data=pd.read_csv("H:\\data\\db2_c.csv")
data1 = pd.read_csv("Mulcross.csv")
data1 = data1.dropna(axis=0)
data2 = pd.read_csv("Mulcross.csv")
data2 = data2.dropna(axis=0)
score = pd.read_csv("Mulcross.csv")
score = score.dropna(axis=0)
# score2=pd.read_csv("H:\\data\\db2_t2.csv")
# fpca=pca.fit_transform(data)
print('data1:', data1)
print('data2:', data2)
print('score:', score)

ifor = IsolationForest(behaviour='new', max_samples=256,
                       random_state=rng, contamination="auto")
'''
#二维化处理与输出
pca=PCA(n_components=3)
fpca1=pca.fit_transform(data1)
fpca2=pca.fit_transform(data2)
ifor.fit(fpca2)
#pred_train = ifor.predict(data1)
pred_test = ifor.predict(fpca2)
print(roc_auc_score(score,pred_test))
'''
ifor.fit(data1)
# pred_train = ifor.predict(data1)
pred_test = ifor.predict(data2)
print(roc_auc_score(score, pred_test))
