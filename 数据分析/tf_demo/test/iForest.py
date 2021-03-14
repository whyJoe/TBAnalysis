# -*-coding:utf-8-*-
"""
Created on 2019/8/20 14:53
@author: joe
"""
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from tf_demo.test import UseData

xtrain,xtest,ytrain,ytest = UseData.data()

ifor = IsolationForest(behaviour='new', max_samples=256,
                       random_state=1, contamination=.05)
ifor.fit(xtrain)
y_pre = ifor.predict(xtest)

print('roc = ', roc_auc_score(ytest, y_pre))

tp, fp, tn, fn = [0, 0, 0, 0]
for i in range(0, len(y_pre)):
    if y_pre[i] == 1 and ytest[i] == 1:
        tp = tp + 1
    elif y_pre[i] == 1 and ytest[i] == 0:
        fp = fp + 1
    elif y_pre[i] == 0 and ytest[i] == 0:
        tn = tn + 1
    else:
        fn += 1
try:
    print('tp : ', tp, 'fp : ', fp, 'tn : ', tn, 'fn : ', fn)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    print('tpr : ', tpr, 'fpr : ', fpr)
    print('ACC : ', (tp + tn) / (tp + tn + fp + fn))
    print('查全率 : ', recall_score(y_true=ytest, y_pred=y_pre))
    # print('异常点准确率 : ', tn / (tn + fn), '查全率 : ', tn / (tn + fp))
except BaseException:
    pass



