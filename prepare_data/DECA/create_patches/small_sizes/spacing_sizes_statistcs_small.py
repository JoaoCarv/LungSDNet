# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

Script creates a csv with only the spacing and nodule's size of
the smaller nodules.

Required changes are:
- path of csv "spacing_sizes_statistics.csv" (path_stats_csv)

"""

import pandas as pd
import csv

path_stats_csv = r''
path_smalls_csv = 'small_sizes.csv'

df_stats_csv = pd.read_csv(path_stats_csv)
df_smalls_csv = pd.read_csv(path_smalls_csv)
print(df_stats_csv)

path_stats_csv_out = 'spacing_sizes_statistics_small.csv'
row = ['numb', 'image_numb', 'spc_x', 'spc_y', 'spc_z', 'sz_x', 'sz_y', 'sz_z']

with open(path_stats_csv_out, 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)
csvFile.close()

for numb in df_smalls_csv.iloc[:, 0]:
    # print(list(df_stats_csv[df_stats_csv['numb'] == numb].iloc[0,:]))
    info = list(df_stats_csv[df_stats_csv['numb'] == numb].iloc[0,:])
    row = [int(info[0]),
           int(info[1]),
           info[2],
           info[3],
           info[4],
           int(info[5]),
           int(info[6]),
           int(info[7])]
    with open(path_stats_csv_out, "a") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
