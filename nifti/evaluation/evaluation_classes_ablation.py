# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    Script to evaluate the results of the ablation study, where 100 models
    were trained. Infered accuracies in the test set are saved in the path_dir

    paths:
        1. path_dir: location of the outputs of the inferences
        2. path_truth_labels: location of the csv with the groundtruth
        3. path_datsplit: location of the dataset split csv file
"""

import pandas as pd
import os
import evaluation_metrics as ev_mtc
#import csv
import numpy as np


# paths geral
path_dir = r''
path_truth_labels = r''
path_datasplit = r''

# load split
df_data_split = pd.read_csv(path_datasplit,
                            header=None,
                            names=['id', 'split'])

# load truth labels
df_truth = pd.read_csv(path_truth_labels,
                       header=None,
                       names=['id', 'class', 'r1', 'r2', 'r3', 'r4'])

# build mask training
df_mask_training = df_data_split['split'] == 'training'


# build mask test
df_mask_test = df_data_split['split'] != 'training'

# ----------- Build y_truth
df_data_split_type = df_data_split['split']
df_truth_classes = df_truth['class']
y_true_train = []
for i in range(df_data_split_type.size):
    if df_data_split_type.iloc[i] == 'training':
        y_true_train.append(df_truth_classes.iloc[i])
y_true_test = []
for i in range(df_data_split_type.size):
    if df_data_split_type.iloc[i] != 'training':
        y_true_test.append(df_truth_classes.iloc[i])


d_out = {'acc': [],
         'iteration': [],
         'label': []
         }

inf_labels = ['inf100', 'inf200', 'inf', 'inf1000']
out_labels = ['100', '200', '500', '1000']

for out in range(0, 100):
    if out == 3:
        break

    # paths specific
    path_iter_dir = os.path.join(path_dir, 'out_'+str(out))

    #d_out['out'].append('out_'+str(out))

    # run for all inf labels
    for inf_label, out_label in zip(inf_labels, out_labels):
        # load predictions for train and test
        inf_file = inf_label + '\_niftynet_out.csv'
        path_pred_labels = os.path.join(path_iter_dir, inf_file)

        df_pred = pd.read_csv(path_pred_labels,
                              header=None,
                              names=['id', 'class'])
        df_pred_train = df_pred[df_mask_training]
        df_pred_test = df_pred[df_mask_test]

        # build y_pred
        y_pred_train = df_pred_train['class'].values.tolist()
        y_pred_test = df_pred_test['class'].values.tolist()

        # --------- Evalutaion metrics

        # predicitve power
        # ev_mtc.predictive_value(y_true_inf, y_pred)
        # accuracy

        d_out['acc'].append(ev_mtc.accuracy(y_true_train, y_pred_train))
        d_out['iteration'].append(out_label)
        d_out['label'].append('training')
        d_out['acc'].append(ev_mtc.accuracy(y_true_test, y_pred_test))
        d_out['iteration'].append(out_label)
        d_out['label'].append('test')


df_out = pd.DataFrame(data=d_out)
df_out.to_csv(path_dir+r'\_eval.csv')
