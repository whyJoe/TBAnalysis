# -*-coding:utf-8-*-
"""
Created on 2019/10/23 14:45
@author: joe
"""
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from tf_demo.test import UseData


xtrain, xtest, ytrain, ytest = UseData.data()


model = LocalOutlierFactor(n_neighbors=20, contamination=0.05,novelty=True)
model.fit(xtrain, ytrain)
y_pre = model.predict(xtest)

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
print('#' * 30)

