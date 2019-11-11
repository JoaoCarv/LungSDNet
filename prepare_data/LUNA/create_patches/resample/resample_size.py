# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomuncações

Script will ressample LUNA images to the median size in
each direction for the lung segmentation. The major steps are:
1. Finds the new spacing:
From the spacing_size_stats.csv gets all the spacings
and finds the median for each axis

2. Applys ressampling:
With the new found spacing, it applys a ressampling to
every image a saves them to a new directory

Required chages are
- path of the directory of the images (dir_images)
- path of the directory of ressampled iamges (dir_out)
"""



import pandas as pd
import os
import SimpleITK as sitk
import numpy as np
from time import time as t_current

import sys
module_path = r'..\..\..\..\my_utilities'
sys.path.add(module_path)

from resample_ops import resample_image
from resample_ops import get_new_spacing




# Directory of dataset <----- CHANGE HERE
Dir_Dataset = r'C:\JFCImportantes\Universidade\Thesis\dataset'
csv_path_read = 'spacing_sizes_stats.csv'


new_spacing = get_new_spacing(csv_path_read)



time = t_current()

# Resample image
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:  # run through all subsets
    #path for images in
    subset = "subset"+str(i)
    dir_images = os.path.join(Dir_Dataset, subset)
    # dir_images = os.path.join(Dir_Dataset, 'images')

    dir_images_out = os.path.join(Dir_Dataset, r'resample')
    print('-------------------', dir_images)

    for file in os.listdir(dir_images):

        # ----------Import image
        filename = os.fsdecode(file)

        if filename.endswith('.nii.gz') and filename[0] != '.':

            path_image = os.path.join(dir_images, filename)

            old_image = sitk.ReadImage(path_image)

            new_image = resample_image(old_image,
                                       new_spacing,
                                       sitk.sitkBSpline)

            # --------- Save new images
            filename_number = filename[6:9]
            filename = 'image_' + str(i) + '_' + filename_number + '.nii.gz'
            path_image_out = os.path.join(dir_images_out, filename)
            sitk.WriteImage(new_image, path_image_out)
            print(filename, t_current()-time)
            time = t_current()
