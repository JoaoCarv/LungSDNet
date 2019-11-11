# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

Script that creates 808*3 2D 64x64 image patches with randomly sized and placed
nodules. Uses functions from custom module my_utilities.synthetic_ops.py
"""


import sys
from time import time
import pandas as pd
import os
import nibabel as nib
import numpy as np
import csv

module_path = r'..\..\my_utilities'
sys.path.insert(0, module_path)
import synthetic_ops
import stats_nodules

# Find the statistics of the nodules sizes
path = r'C:\JFCImportantes\Universidade\Thesis\code\detection\naive\prep_data\resample\nodules_sizes.csv'
df_nodules_sizes = pd.read_csv(path)

s_nodules_sizes = pd.concat([df_nodules_sizes['nodule size y (vx)'],
                             df_nodules_sizes['nodule size x (vx)']])

stats_dic = stats_nodules.stats_nodules_size(s_nodules_sizes)


Size = 64
max_size = int(stats_dic['max'])
min_size = int(stats_dic['min'])


# Define paths
dir_out = r'C:\JFCImportantes\Universidade\Thesis\dataset\LUNA16\test'
# os.mkdir(dir_out)
dir_images_out = os.path.join(dir_out, 'patches')
# os.mkdir(dir_images_out)
csv_path = os.path.join(dir_images_out, 'nodules_class_ids_synthetic.csv')

t_current = time()

# N_images = 808*3
N_images = 2
s = synthetic_ops.sizes_gamma_distribution(N_images, max_size, min_size)

for i in range(N_images):
    print(i)

    # Create
    a = s[i]
    image_i, x0, y0, a, b = synthetic_ops.elipsoid_random_2d(Size,
                                                             max_size,
                                                             min_size,
                                                             return_origin=True,
                                                             ab_given=True,
                                                             a=a)

    rand_i = np.zeros((Size, Size))

    # Filter image
    filtered_image = synthetic_ops.give_noise(image_i)
    filtered_rand = synthetic_ops.give_noise(rand_i)

    # transform to nifti
    name_image = '0'*(4-len(str(i+1))) + str(i+1)
    name_rand = '0'*(4-len(str(i+1))) + str(i+1) + '_rand'

    # save image
    img_out = nib.Nifti1Image(filtered_image, None)
    rand_out = nib.Nifti1Image(filtered_rand, None)
    # img = nib.Nifti1Image(image_i,None)

    path_image_out = os.path.join(dir_images_out, name_image+'.nii.gz')
    nib.save(img_out, path_image_out)
    with open(csv_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([name_image, 1, x0, y0, a, b])

    path_rand_out = os.path.join(dir_images_out, name_rand + '.nii.gz')
    nib.save(rand_out, path_rand_out)
    with open(csv_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([name_rand, 0, 0, 0, 0, 0])

    print(time()-t_current)
    t_current = time()
