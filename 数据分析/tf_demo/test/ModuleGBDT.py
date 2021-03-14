# -*-coding:utf-8-*-
"""
Created on 2019/11/1 11:20
@author: joe
"""

from sklearn.ensemble import GradientBoostingClassifier
from tf_demo.test import UseData
import pandas as pd

x, x_test, y, y_test = UseData.data()

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(
    loss='deviance',  ##损失函数默认deviance  deviance具有概率输出的分类的偏差
    n_estimators=100,  ##默认100 回归树个数 弱学习器个数
    learning_rate=0.1,  ##默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长
    max_depth=5,  ## 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
    subsample=1,  ##树生成时对样本采样 选择子样本<1.0导致方差的减少和偏差的增加
    min_samples_split=2,  ##生成子节点所需的最小样本数 如果是浮点数代表是百分比
    min_samples_leaf=1,  ##叶节点所需的最小样本数  如果是浮点数代表是百分比
    max_features=None,  ##在寻找最佳分割点要考虑的特征数量auto全选/sqrt开方/log2对数/None全选/int自定义几个/float百分比
    max_leaf_nodes=None,  ##叶节点的数量 None不限数量
    min_impurity_split=1e-7,  ##停止分裂叶子节点的阈值
    verbose=0,  ##打印输出 大于1打印每棵树的进度和性能
    warm_start=False,  ##True在前面基础上增量训练(重设参数减少训练次数) False默认擦除重新训练
    random_state=0  ##随机种子-方便重现
).fit(x, y)  ##多类别回归建议使用随机森林
print('score : ', clf.score(x_test, y_test))  ##tp / (tp + fp)正实例占所有正实例的比例
y_pre = clf.predict(x_test)
from sklearn import metrics
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
# fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pre)
print('tp : ', tp, 'fp : ', fp, 'tn : ', tn, 'fn : ', fn)
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
print('tpr : ', tpr, 'fpr : ', fpr)
print('ACC : ', (tp + tn) / (tp + tn + fp + fn))
print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pre))
