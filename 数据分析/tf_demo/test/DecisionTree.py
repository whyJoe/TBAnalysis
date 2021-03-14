# -*-coding:utf-8-*-
"""
Created on 2019/10/22 17:22
@author: joe
"""
from tf_demo.test import UseData
from sklearn import tree
import pandas as pd
from sklearn.metrics import recall_score

xtrain,xtest,ytrain,ytest = UseData.data()

clf = tree.DecisionTreeClassifier(random_state=800)
clf = clf.fit(xtrain, ytrain)
y_pre = clf.predict(xtest)
score = clf.score(xtest, ytest)

print('score : ', score)

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
y_pre = pd.DataFrame(y_pre.reshape(-1, 1))
print('预测正负例的数量 : ', y_pre.iloc[:, -1].value_counts())

