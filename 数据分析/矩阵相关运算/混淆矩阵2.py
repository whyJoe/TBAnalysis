# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix,roc_curve,auc,precision_recall_curve,average_precision_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


def confusion_metrix(y,y_p):
    Confusion_matrix = confusion_matrix(y,y_p)
    plt.matshow(Confusion_matrix)
    plt.title("混淆矩阵")
    plt.colorbar()
    plt.ylabel("实际类型")
    plt.xlabel("预测类型")



