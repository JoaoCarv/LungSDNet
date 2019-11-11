# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    accuracy(y_true,y_pred):
        Returns the accuracy of the prediction.
    predictive_value(y_true, y_pred):
        Returns the positive and negative predictive power.
    roc_curve(y_true, y_pred):
        Returns the Receiver Operating Characteristic.
    auc_roc_curve(y_true, y_pred):
        Returns the Area Under the Curve for the ROC curve.
"""
from sklearn import metrics
import matplotlib.pyplot as plt


def accuracy(y_true, y_pred):
    """
    Returns the accuracy of the prediction.
    The second output takes into account imbalacenments in
    the dataset. As such it is defined as the average of recall
    obtained on each class.

    Args:
    y_true (array-like): ground truth
    y_pred (array-like): probability estimates predicted by the model

    returns 1 float: accuracy
    (optionally it returns the acc taking into account the dataset imb)
    """
    y_pred = [round(x) for x in y_pred]
    normalization = True
    accuracy = metrics.accuracy_score(y_true, y_pred, normalize=normalization)
    b_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
    return accuracy #, b_accuracy


def predictive_value(y_true, y_pred):
    """
    Returns the positive and negative predictive power.

    Args:
    y_true (array-like): ground truth
    y_pred (array-like): probability estimates predicted by the model

    returns 2 floats: positive and false predictive power
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if y_pred[i] == 0:
                TN += 1
            else:
                FP += 1

    # print('TP: ', TP)
    # print('FP: ', FP)
    # print('TN: ', TN)
    # print('FN: ', FN)

    ppv = TP/(TP+FP)
    fpv = TN/(TN+FN)
    # print('predictive',ppv,fpv)
    return ppv, fpv


def roc_curve(y_true, y_pred):
    """
    Returns the Receiver Operating Characteristic.
    Outputs:
    - fpr: false positve rates in increasing order, such that the
    element i is the false positive rate of predictions with score
    larger than thresholds[i]
    - tpr: same as fpr with the true positive rate

    Args:
    y_true (array-like): ground truth
    y_pred (array-like): probability estimates predicted by the model

    returns 2 arrays
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)

    return fpr, tpr


def plot_roc_curve(fpr, tpr):
    """

     Function for ploting the curves.
     Args:
     fpr (array): false positive rates in increasing order
     tpr (array): true positive rates in increasing order

    """
    plt.plot(fpr, tpr, color='cornflowerblue', label='ROC')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    #plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend()
    plt.show()


def auc_roc_curve(y_true, y_pred):

    """
    Returns the Area Under the Curve for the ROC curve.

    Args:
    y_true (array-like): ground truth
    y_pred (array-like): probability estimates predicted by the model

    returns a float
    """
    auc = metrics.roc_auc_score(y_true, y_pred)
    return auc
