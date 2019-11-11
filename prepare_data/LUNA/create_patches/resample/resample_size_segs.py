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
# from radiomics.imageoperations import resampleImage
import SimpleITK as sitk
import numpy as np

import sys
module_path = r'..\..\..\..\my_utilities'
sys.path.insert(0, module_path)
import luna_masks_ops as mops

from resample_ops import resample_image
from time import time as t_current

new_spacing = [0.0703000009059906,
               0.0703000009059906,
               1.25]  # obtained from the median spacing of the images


time = t_current()

dir_images = r''
dir_images_out = r''

for file in os.listdir(dir_images):

    # ----------Import image
    filename = os.fsdecode(file)

    if filename.endswith('.nii.gz') and filename[0] != '.':

        path_image = os.path.join(dir_images, filename)

        old_image = sitk.ReadImage(path_image)

        new_image = resample_image(old_image,
                                   new_spacing,
                                   sitk.sitkNearestNeighbor)
                                   #sitk.sitkBSpline)

        # --------- Save new images
        # filename_number = filename[6:9]
        # filename = 'image_' + str(i) + '_' + filename_number + '.nii.gz'
        path_image_out = os.path.join(dir_images_out, filename)
        sitk.WriteImage(new_image, path_image_out)
        print(filename, t_current()-time)
        time = t_current()
