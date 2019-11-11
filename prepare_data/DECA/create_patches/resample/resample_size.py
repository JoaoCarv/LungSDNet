"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomuncações

Script will ressample Decathlon daset images to the median size in
each direction. The major steps are:
1. Finds the new spacing:
From the spacing_size_stats.csv gets all the spacings
and finds the median for each axis

2. Applys ressampling:
With the new found spacing, it applys a ressampling to
every image a saves them to a new directory

paths:
Only required changes are in the path of the directory
of the images (Dir_Dataset)
"""



import pandas as pd
import os
# from radiomics.imageoperations import resampleImage
import SimpleITK as sitk
import numpy as np
from time import time as t_current

import sys
module_path = r'..\..\..\..\my_utilities'
sys.path.add(module_path)

from resample_ops import resample_image


Dir_Dataset = r''
dir_images = os.path.join(Dir_Dataset, 'images')
dir_masks = os.path.join(Dir_Dataset, 'masks')
csv_path_read = '../spacing_sizes_stats.csv'

dir_images_out = os.path.join(Dir_Dataset, r'resample\images')
dir_masks_out = os.path.join(Dir_Dataset, r'resample\masks')

# new_spacing = get_new_spacing(csv_path_read)


new_spacing = [0.785, 0.785, 1.245]
new_spacing = np.array(new_spacing)

time = t_current()

# Resample image
for file in os.listdir(dir_images):

    # ----------Import image
    filename = os.fsdecode(file)

    if filename.endswith('.nii.gz') and filename[0] != '.':

        path_image = os.path.join(dir_images, filename)
        path_mask = os.path.join(dir_masks, filename)

        old_image = sitk.ReadImage(path_image)
        old_mask = sitk.ReadImage(path_mask)

        # --------- Resample
        # new_image, new_mask = resampleImage(old_image, old_mask,
        #                                     resampledPixelSpacing=new_dims,
        #                                     preCrop=False)

        new_image = resample_image(old_image,
                                   new_spacing,
                                   sitk.sitkBSpline)
        new_mask = resample_image(old_mask,
                                  new_spacing,
                                  sitk.sitkNearestNeighbor)

        # --------- Save new images
        path_image_out = os.path.join(dir_images_out, filename)
        path_mask_out = os.path.join(dir_masks_out, filename)
        sitk.WriteImage(new_image, path_image_out)
        sitk.WriteImage(new_mask, path_mask_out)
        print(filename, t_current()-time)
        time = t_current()
