# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    evaluate the classification model's results.
    Metrics evaluated:
        - true positives
        - false positives
        - ROC curve and AUC
    paths:
        dir_models: directory of the model (where the inference is)
        dir_labels: directory where the ground truth csv is.
        dir_split: path of the csv file with the dataset split
        dir_out: path to the output file

"""

import pandas as pd
import os
import evaluation_metrics as ev_mtc
import csv
import numpy as np

# paths
dir_model = r''
dir_labels = r''

path_data_split = r''
path_pred = os.path.join(dir_model, '_niftynet_out.csv')
path_truth = os.path.join(dir_labels, 'nodules_class.csv')


# Load all files
df_data_split = pd.read_csv(path_data_split,
                            header=None,
                            names=['id', 'split'])
df_pred = pd.read_csv(path_pred,
                      header=None,
                      names=['id', 'class'])

df_truth = pd.read_csv(path_truth,
                       header=None,
                       names=['id', 'class', 'r1', 'r2', 'r3', 'r4'])

# ----------- Build y_truth
df_data_split_type = df_data_split['split']
df_truth_classes = df_truth['class']
y_true = []
for i in range(df_data_split_type.size):
    if df_data_split_type.iloc[i] == 'inference':
        y_true.append(df_truth_classes.iloc[i])

# ---------- Build y_pred
y_pred = df_pred['class'].values.tolist()


# --------- Evalutaion metrics
out = []
header = ['acc', 'ftp', 'tpr', 'auc']
# predicitve power
ev_mtc.predictive_value(y_true, y_pred)
# accuracy
out.append(ev_mtc.accuracy(y_true, y_pred))
# roc_curve
fpr, tpr = ev_mtc.roc_curve(y_true, y_pred)
out.append(fpr)
out.append(tpr)
# auc roc
out.append(ev_mtc.auc_roc_curve(y_true, y_pred))


path_csv_out = r''

with open(path_csv_out, 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(header)
    writer.writerow(out)
csvFile.close()

# -------- Plot ROC Curve
ev_mtc.plot_roc_curve(fpr, tpr)
