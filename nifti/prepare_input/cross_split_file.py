# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    Script that separates the datset into
    K groups of images and then creates a split file for
    the images to be fed to niftynet
"""
import os
import csv
import random

K = 5

# dir_im = r'D:\JFC_Lung\dataset\resample\big\more_non\images'
dir_im = r'C:\JFCImportantes\Universidade\Thesis\dataset\resample\images'
dir_out = r''
# make list of files
list_files = []

for file in os.listdir(dir_im):
    filename = os.fsdecode(file)
    list_files.append(filename)

list_files_numb = [x[5:8] for x in list_files]


def kfoldcv(indices, k = K, seed = None):

    size = len(indices)
    subset_size = round(size / k)
    random.Random(seed).shuffle(indices)
    subsets = [indices[x:x+subset_size] for x in range(0, len(indices), subset_size)]
    kfolds = []
    for i in range(k):
        test = subsets[i]
        train = []
        for subset in subsets:
            if subset != test:
                train.append(subset)
        kfolds.append((train,test))

    return kfolds


cv_lists = kfoldcv([x for x in range(len(list_files))])
# print(cv_lists)
for k in range(K):

    csv_filename = 'split_data' + str(k+1) + '.csv'
    csv_path = os.path.join(dir_out, csv_filename)
    for i in range(len(list_files)):
        if i in cv_lists[k][1]:
            row = [list_files_numb[i], 'validation']
        else:
            row = [list_files_numb[i], 'training']

        with open(csv_path, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
