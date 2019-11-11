"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

Script goes through all the images from the dataset and appends
its sizes and spacings to a csv file.

Required changes are:
- path of the directory of the images (DirectoryOfDataset)
"""

import sys
import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import csv


# Directory
DirectoryOfDataset = r'D:\JFC_Lung\LUNA16\NIFTI_2'


# CSV
csv_path = r'spacing_sizes_stats.csv'
csv_path = os.path.join(DirectoryOfDataset, csv_path)
row = ['numb', 'image_numb', 'spc_x', 'spc_y', 'spc_z', 'sz_x', 'sz_y', 'sz_z']
with open(csv_path, 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)

    image_numb = 0
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:  # run through all subsets
        #path for images in
        subset = "subset"+str(i)
        dir_images_in = os.path.join(DirectoryOfDataset, subset)

        print('-------------------', dir_images_in)

        for file in os.listdir(dir_images_in):
            # if image_numb > 1:
            #     exit()

            filename = os.fsdecode(file)

            if filename.endswith('.nii.gz') and filename[0] != '.':

                filename_number = filename[6:9]

                path_image_in = os.path.join(dir_images_in, filename)
                image_sitk = sitk.ReadImage(path_image_in)

                Spacing = list(image_sitk.GetSpacing())
                Size = list(image_sitk.GetSize())

                # Write to CSV file
                row = [image_numb, filename_number, Spacing[0], Spacing[1],
                       Spacing[2], Size[0], Size[1], Size[2]]

                # Write to csv file
                writer.writerow(row)



                print(image_numb, ': ', filename)
                image_numb += 1
                print('------------')
csvFile.close()
