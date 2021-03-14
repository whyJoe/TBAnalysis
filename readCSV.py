# -*-coding:utf-8-*-
"""
Created on 2021/3/14 14:45
@author: joe
"""

import pandas as pd
import matplotlib.pyplot as plt
import tool
import numpy as np
# 导入引擎
from pyecharts import options
# 导入柱状图
from pyecharts.charts import Bar
from collections import Counter

# 读取数据
data = pd.read_csv('train_user.csv')
# 测试数据从前1000条中提取
data = data.iloc[:1000, :]
print(data)

# 日访问量分析(Analysis of daily visits)
def dailyAnalysis(data):
    dateData = []
    countData = []
    for i in data['time']:
        dateData.append(tool.getDate(i))
    # 对日期进行排序
    dateData = sorted(dateData)
    unique_data = np.unique(dateData)
    # 统计次数
    for i in unique_data:
        countData.append(dateData.count(i))
    plt.plot(unique_data, countData)
    plt.xticks(rotation=45)
    plt.title('Analysis of daily visits', fontsize=14)
    plt.grid()
    plt.show()

dailyAnalysis(data)



# 不同行为类型用户分析(Analysis of users with different behavior types)
def differentAnalysis(data):
    return 0


# 用户购买情况次数分析(Analysis of user purchase)
def purchaseAnalysis(data):
    itemData = []
    countData = []
    for i in data['item_id']:
        itemData.append(i)

    unique_data = np.unique(itemData)
    # 统计次数
    for i in unique_data:
        countData.append(itemData.count(i))
    print(unique_data, countData)
    return 0
purchaseAnalysis(data)

# 日PV(Daily PV)
def dailyPV(data):
    return 0;


# 日人均PV(Daily per capita PV)
def DailyPerPV():
    return 0


# 日UV(Daily UV)
def dailyUV(data):
    return 0


# 付费率(RATE)
def RATE(data):
    return 0


# 同一时间段用户消费次数分布(Distribution of consumption times of users in the same period)
def samePeriodDistribution(data):
    return 0
