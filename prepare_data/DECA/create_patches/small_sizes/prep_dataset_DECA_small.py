# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

Script for pre-processing each image. The steps are:
1. Apply padding
2. Define bounding box
3. Write new patches

Required changes are:
- path of the directory of the images (DirectoryOfDataset)
- path of the out directory (DirectoryOfOutput)

Note that size of patch can be redefined in "Max_Size"
"""

import sys
import os
import SimpleITK as sitk
import pandas as pd

sys.path.append("../../../../my_utilities")
import utilities as utl
from create_patches import deca_create_3d_patches

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# Directories path
DirectoryOfDataset = r''
DirectoryOfOutput = r''
os.mkdir(DirectoryOfOutput)


# Paths to images
dir_images_in = os.path.join(DirectoryOfDataset, 'images')
dir_masks_in = os.path.join(DirectoryOfDataset, 'masks')

dir_images_out = os.path.join(DirectoryOfOutput, 'images')
os.mkdir(dir_images_out)
dir_masks_out = os.path.join(DirectoryOfOutput, 'masks')
os.mkdir(dir_masks_out)

# PATCH Size
Max_size = [64, 64]  # 121, 146, 86
sz2 = int(Max_size[0]/2)


# CSVs for sizes and spacings
csv_sizes_small = 'spacing_sizes_statistics_small.csv'
pandas_rs_sizes_small = pd.read_csv(csv_sizes_small)
rs_size_z = pandas_rs_sizes_small['sz_z']
csv_path = 'small_sizes.csv'
pandas_numbers = pd.read_csv(csv_path)
pandas_numbers = pandas_numbers['image_name']


image_numb = 0
sizes_n = 0
for file in os.listdir(dir_images_in):
    if image_numb > 7:
        exit()

    filename = os.fsdecode(file)

    if filename.endswith('.nii.gz') and filename[0] != '.':
        print('---', image_numb)
        image_numb += 1

        if pandas_numbers[pandas_numbers.isin([filename])].count() > 0:

            # Load image
            filenumber = filename[5:-11]
            path_image_in = os.path.join(dir_images_in, filename)
            path_mask_in = os.path.join(dir_masks_in, filename)

            sitk_image = sitk.ReadImage(path_image_in)
            sitk_mask = sitk.ReadImage(path_mask_in)

            # Apply paddig
            padding = [100, 100, 100]
            pad_filter = sitk.ConstantPadImageFilter()
            pad_filter.SetPadLowerBound(padding)
            pad_filter.SetPadUpperBound(padding)
            pad_filter.SetConstant(-1000.0)
            sitk_image = pad_filter.Execute(sitk_image)
            sitk_mask = pad_filter.Execute(sitk_mask)

            # Setting Bounding Box
            F_statistics = sitk.LabelShapeStatisticsImageFilter()
            F_statistics.Execute(sitk_mask)
            bbox_dims = F_statistics.GetBoundingBox(1)  # only 1 label /p image

            patch_image, patch_mask = deca_create_3d_patches(bbox_dims,
                                                             rs_size_z,
                                                             Max_size,
                                                             sizes_n,
                                                             sitk_image,
                                                             sitk_mask)

            path_image_out = os.path.join(dir_images_out, filename)
            path_mask_out = os.path.join(dir_masks_out, filename)

            sitk.WriteImage(patch_image, path_image_out)
            sitk.WriteImage(patch_mask, path_mask_out)



utl.beep_sound()
