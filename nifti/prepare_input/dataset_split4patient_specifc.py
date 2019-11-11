# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    prepare dataset-split file for hold out testing, taking into account
    patient separation. 80% of the dataset for training, 10% for validation,
    10% for testing.
"""

import os
import csv
path_dir = r'D:\JFC_Lung\LUNA16\NIFTI_2\resample\Nvar'
path_csv = os.path.join(path_dir, 'dataset_split_Nvar.csv')
print(path_csv)


# Training set
path_tr_dir = os.path.join(path_dir, 'patches_train')
for file in os.listdir(path_tr_dir):
    filename = os.fsdecode(file)
    print(filename)
    row = [filename[6:-7], 'training']
    with open(path_csv, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()


# Validation set
path_vld_dir = os.path.join(path_dir, 'patches_val')
for file in os.listdir(path_vld_dir):
    filename = os.fsdecode(file)
    row = [filename[6:-7], 'validation']
    with open(path_csv, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()

# Inference set
path_inf_dir = os.path.join(path_dir, 'patches_infer')
for file in os.listdir(path_inf_dir):
    filename = os.fsdecode(file)
    row = [filename[6:-7], 'inference']
    with open(path_csv, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
