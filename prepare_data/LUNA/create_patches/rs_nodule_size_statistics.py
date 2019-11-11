# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

Script goes through all the resampled images from the
dataset of LUNA16 and appends the name, the nodule size
in mm, and its size in voxels for each direction (nodules_sizes.csv)


Required changes are:
- path to the raw dataset (Directory_raw)
- path to the resampled images (Directory_resampled)
"""


import pandas as pd
import os
import csv
import numpy as np

# Define directory
Directory_raw = r''
Directory_resampled = r''

# csv files
annotations_path = os.path.join(Directory_raw, 'annotations.csv')
annotations_df = pd.read_csv(annotations_path)

names_convert_path = os.path.join(Directory_raw, 'names_convert.csv')
names_convert_df = pd.read_csv(names_convert_path)

old_spacing_path = os.path.join(Directory_resampled, 'spacing_sizes_stats.csv')
old_spacing_df = pd.read_csv(old_spacing_path)

new_spacing_path = os.path.join(Directory_resampled, 'spacing_sizes_stats.csv')
new_spacing_df = pd.read_csv(new_spacing_path)
new_spacing = [new_spacing_df.iloc['spc_x'].iloc[0].values,
               new_spacing_df['spc_y'].iloc[0].values,
               new_spacing_df['spc_z'].iloc[0].values]

# Prepare new csv_file
new_sizes_csv_path = os.path.join(Directory_resampled, 'nodules_sizes.csv')
with open(new_sizes_csv_path, 'a') as csvFile:
    writer = csv.writer(csvFile)
    row = ['name', 'nodule_size (mm)',
           'nodule size x (vx)',
           'nodule size y (vx)',
           'nodule size z (vx)',
           'origin x',
           'origin y',
           'origin z']

    writer.writerow(row)
csvFile.close()


# Functions auxiliary
def mm2voxel(spacing, mm):
    spacing = np.array(spacing)
    mm = np.array(mm)
    return mm/spacing


def get_nodules_info(annotations_df, names_convert_df, new_name):
    name_seriesuid = names_convert_df.loc[names_convert_df['newname'] == new_name]
    name_seriesuid = name_seriesuid.iloc[0, 0]

    nodules_info = annotations_df.loc[annotations_df['seriesuid'] == name_seriesuid]
    return nodules_info


# --------- Calculate new sizes for each image
# Number of images
n_total = int(names_convert_df.size/2)
# Get images_identifiers
images_identifiers = names_convert_df['newname']
for image_i in range(n_total):
    image_new_name = images_identifiers.iloc[image_i]

    # Get information regarding nodules in image
    nodule_info = get_nodules_info(annotations_df,
                                   names_convert_df,
                                   image_new_name)

    n_nodules = nodule_info.shape[0]
    for nodule_i in range(n_nodules):
        # Calculate new size for each nodule
        mm_size = nodule_info['diameter__mm'].iloc[nodule_i]
        new_size = mm2voxel(new_spacing, mm_size)

        # save to csv
        with open(new_sizes_csv_path, 'a') as csvFile:
            writer = csv.writer(csvFile)
            nx = new_size[0]
            ny = new_size[1]
            nz = new_size[2]
            row =[image_new_name, mm_size,
                  nx, ny, nz]
            writer.writerow(row)
        csvFile.close()
