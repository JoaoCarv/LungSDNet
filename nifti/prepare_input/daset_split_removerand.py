# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    prepare dataset-split file for hold out testing. 80% of the dataset for
    training, 10% for validation, 10% for testing.
"""


import os
import csv
path_dir = r''
path_csv = r''

# Training set
for file in os.listdir(path_dir):
    filename = os.fsdecode(file)
    if int(filename[6]) < 8:
        filenamen = os.fsdecode(file)
        row = [filename[6:14], 'training']
        with open(path_csv, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()


# Validation set
for file in os.listdir(path_dir):
    filename = os.fsdecode(file)
    if int(filename[6]) == 8:
        row = [filename[6:14], 'validation']
        with open(path_csv, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()

# Inference set
for file in os.listdir(path_dir):
    filename = os.fsdecode(file)
    if int(filename[6]) == 9:
        row = [filename[6:14], 'inference']
        with open(path_csv, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
