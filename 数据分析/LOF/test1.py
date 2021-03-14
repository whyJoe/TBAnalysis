# -*-coding:utf-8-*-
"""
Created on 2019/8/21 16:12
@author: joe
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

# Mulcross原图
# data = pd.read_csv('Mulcross.csv')
# estimator = PCA(n_components=2)
# data1 = estimator.fit_transform(data)
# plt.scatter(data1[:,0],data1[:,1],c=data.iloc[:,4])
# plt.show()


# db2原图
# data = pd.read_csv('db2.csv')
# estimator = PCA(n_components=2)
# data1 = estimator.fit_transform(data.iloc[:,:5])
# plt.scatter(data1[:,0],data1[:,1],c=data.iloc[:,5])
# plt.show()

# haberman原图
data = pd.read_csv('haberman.csv')
data.columns=['age','f1','f2','class']
# sns.barplot(x='class',y='f1',data=data)
estimator = PCA(n_components=2)
data1 = estimator.fit_transform(data.iloc[:,:3])
plt.title('Haberman')
plt.scatter(data1[:,0],data1[:,1],c=data.iloc[:,3])
plt.show()
