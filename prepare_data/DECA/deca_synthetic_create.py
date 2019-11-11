# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    Makes 64 synthetic 3D images with randomly sized and placed nodules.
    Uses functions from custom module my_utilities.synthetic_ops.py
"""


import sys
from time import time
import os
import nibabel as nib

module_path = r'..\..\my_utilities'
sys.path.insert(0, module_path)
import synthetic_ops

Size = 150
max_size = 140
min_size = 30

dir_out = r'C:\JFCImportantes\Universidade\Thesis\dataset\synthetic'
dir_images_out = os.path.join(dir_out, 'images')
dir_masks_out = os.path.join(dir_out, 'masks')

t_current = time()
for i in range(64):

    # Create raw image with synthetic nodule
    image_i = synthetic_ops.elipsoid_random_3d(Size,
                                               max_size,
                                               min_size)
    # Filter image
    filtered_image = synthetic_ops.give_noise(image_i, gaussian_noise=False)

    # transform to nifti
    name = 'image_' + '0'*(2-len(str(i+1))) + str(i+1)

    img_out = nib.Nifti1Image(filtered_image, None)
    mask_out = nib.Nifti1Image(image_i, None)
    # img = nib.Nifti1Image(image_i,None)

    path_image_out = os.path.join(dir_images_out, name+'_bar.nii.gz')
    path_mask_out = os.path.join(dir_masks_out, name+'_bar.nii.gz')
    nib.save(img_out, path_image_out)
    nib.save(mask_out, path_mask_out)
    print(time()-t_current)
    t_current = time()
