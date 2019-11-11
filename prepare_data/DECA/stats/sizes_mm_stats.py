# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

Script goes through all the images from the dataset and appends the
sizes of the nodules in mm to the datase

Required changes are:
- path of the directory of the images (DirectoryOfDataset)
- path of csv (csv_path) where the file "sizes_mm_stats.csv" is.
"""


import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import csv
import sys
module_path = r'..\..\my_utilities'
sys.path.append(module_path)
import utilities
import sitk_ops

# Directory
DirectoryOfDataset = r''


# Paths to images
# dir_images_in = os.path.join(DirectoryOfDataset, 'images_patches')
dir_images_in = os.path.join(DirectoryOfDataset, 'masks')

# CSV
csv_path = r'sizes_mm_stats.csv'
row = ['numb', 'image_name', 'sz_x', 'sz_y', 'sz_z']
with open(csv_path, 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)
csvFile.close()

image_numb = 0
# max_size = [0, 0, 0]
# Spacing_mm_l = np.array([[0.0,0.0,0.0]])

for file in os.listdir(dir_images_in):

    filename = os.fsdecode(file)

    if filename.endswith('.nii.gz') and filename[0] != '.':

        image_numb += 1
        filename_number = filename[5:8]

        path_image_in = os.path.join(dir_images_in, filename)
        image_sitk = sitk.ReadImage(path_image_in)

        Size = np.array(sitk_ops.bbox_size(image_sitk))

        Spacing = np.array(list(image_sitk.GetSpacing()))
        Size_mm = Size*Spacing


        # Write to CSV file
        row = [image_numb, filename, Size_mm[0], Size_mm[1], Size_mm[2]]


        utilities.append2csv(csv_path, row)

        print('---------------------')

        print(image_numb, ': ', filename)
        #print(image_numb, ': ', Size)
        print(row)


size_mm = pd.DataFrame(Spacing_mm_l)
size_mm.hist(bins=7)

utl.beep_sound()
