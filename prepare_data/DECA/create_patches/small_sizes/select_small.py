# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

Script goes through all the images from the dataset and appends the
sizes of the nodules in mm to the datase selecting only the ones between
the "lower_limit" and the "upper_limit".

Required changes are:
- path of csv "sizes_mm_stats.csv" (path_sizes_csv)
- path of the directory of the images (DirectoryOfDataset)

"""

import pandas as pd
import csv
import os

DirectoryOfDataset = r''
dir_images_in = os.path.join(DirectoryOfDataset, 'masks')
lower_limit = 25  # 30 mm -> from article
upper_limit = 60  # 60 mm -> so that they can fit into 64x64 patches

path_sizes_csv = r''

path_out_csv = r'small_sizes.csv'
row = ['numb', 'image_name']
with open(path_out_csv, 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)
csvFile.close()


df_sizes_mm = pd.read_csv(path_sizes_csv)


image_numb = 1
for file in os.listdir(dir_images_in):

    filename = os.fsdecode(file)

    if filename.endswith('.nii.gz') and filename[0] != '.':


        Sizes = list(df_sizes_mm.iloc[image_numb-1,2::])

        # test lower bound (only one direction has to pass)
        lower_bound_test = any(list(map(lambda x: x>lower_limit, Sizes)))

        # test upper bound (all directions need to pass)
        upper_bound_test = all(list(map(lambda x: x<upper_limit, Sizes)))

        if lower_bound_test is True and upper_bound_test is True:
            row = [image_numb, filename]
            with open(path_out_csv, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()

        image_numb += 1
        print(image_numb)
