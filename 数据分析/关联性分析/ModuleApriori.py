# -*-coding:utf-8-*-
"""
Created on 2019/12/9 9:37
@author: joe
"""
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# 设置数据集
dataset = [['牛奶', '洋葱', '肉豆蔻', '芸豆', '鸡蛋', '酸奶'],
           ['莳萝', '洋葱', '肉豆蔻', '芸豆', '鸡蛋', '酸奶'],
           ['牛奶', '苹果', '芸豆', '鸡蛋'],
           ['牛奶', '独角兽', '玉米', '芸豆', '酸奶'],
           ['玉米', '洋葱', '洋葱', '芸豆', '冰淇淋', '鸡蛋']]

te = TransactionEncoder()
# 进行one-hot编码
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
# 利用Apriori
freq = apriori(df,min_support=0.05,use_colnames=True)

print(freq)