# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

Script that creates a segmentation mask following spheric shape aroudn the
lesion.

Required changes are:
- path of the directory of the images (Dir_Dataset)
"""

import pandas as pd
import os
import SimpleITK as sitk
import nnumpy as np

import sys
module_path = r'..\..\my_utilities'
sys.path.insert(0, module_path)
import luna_masks_ops as mops


# Directory
Dir_Dataset = r''
dir_images_in = os.path.join(Dir_Dataset, r'resample\images')
dir_images_out = os.path.join(Dir_Dataset, r'resample\masks')

# csvs
# nodules sizes
nodules_sizes_path = os.path.join(Dir_Dataset, r'resample\nodules_sizes.csv')
nod_size_df = pd.read_csv(nodules_sizes_path)
im_considered = nod_size_df['name']

# origin
annotations_path = os.path.join(Dir_Dataset, 'annotations.csv')
annotations_df = pd.read_csv(annotations_path)

names_convert_path = os.path.join(Dir_Dataset, 'name_convert.csv')
names_convert_df = pd.read_csv(names_convert_path)

for file in os.listdir(dir_images_in):

    filename = os.fsdecode(file)
    print('---------', filename)
    if filename.endswith('.nii.gz') and filename[0] != '.':

        filename_no_ext = os.path.splitext(filename[:-3])[0]
        image_path = os.path.join(dir_images_in, filename)

        # Check if image is going to be used (according to LUNA16 dataset)
        # = check if exists in nodules_sizes (b true doesnt exist)
        b = im_considered[im_considered.isin([filename_no_ext])].empty

        if b is False:
            # Load image
            image = sitk.ReadImage(image_path)

            # Get inforamtion from image
            Origin = list(image.GetOrigin())
            Spacing = list(image.GetSpacing())
            Size = list(image.GetSize())

            # Get nodules size and origin
            nodules_out = mops.get_nodules_info(annotations_df,
                                           names_convert_df,
                                           filename_no_ext,
                                           Origin,
                                           nod_size_df)

            # Create image
            image_nda = np.zeros(Size[::-1])
            for nodule in nodules_out:
                a = nodule[3]
                b = nodule[4]
                c = nodule[5]

                image_nda_nodule = mops.elipsoid(Size, a, b, c, nodule[0:3])
                image_nda = image_nda + image_nda_nodule

            mask_sitk = sitk.GetImageFromArray(image_nda)
            mask_sitk.CopyInformation(image)


            # Save mask
            mask_out_path = os.path.join(dir_images_out, filename)
            sitk.WriteImage(mask_sitk, mask_out_path)
