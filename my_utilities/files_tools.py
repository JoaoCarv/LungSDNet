# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    count_csv(dir_in, name):
        count the number of objects in hte csv
    mhd2nifiti(file_name, dir_in, dir_out, img_nb, write=True):
        load .mhd file to sitk (returning it) and writing it

"""

import os
import pandas as pd
import SimpleITK as sitk


def count_csv(dir_in, name):
    """ Function for counting the number of elements in the csv.

    Parameters:
        dir: (str) path to the directory where the csv is
        name: (str) name of the csv

    Returns:
        (int) number of elements
    """
    csv_path = os.path.join(dir_in, name)
    csv_data = pd.read_csv(csv_path)
    return csv_data.shape[0]


def mhd2nifti(file_name, dir_in, dir_out, img_nb, write=True):
    """ Function for converting a .mhd image to .nii.gz and
    write it.

    Parameters:
        filename: (str) name of the csv
        dir_in: (str) path to the directory where the .mhd image is
        dir_out: (str) path to the directory where the .nii.gz image will be
            writen to
        img_nb: (str) number of the image
        write: (boolean) True means that the image will be writen


    Returns:
        sitk_image (SITK Image) converted image in SITK format
    """

    file = os.path.join(dir, file_name)
    filename = os.fsdecode(file)
    if filename.endswith(".mhd"):
        out_filename = os.path.join(
            dir_out, "image" + "0" * (3 - len(str(img_n))) + str(img_nb)
        )

        dir_out = out_filename + ".nii.gz"
        sitk_image = sitk.ReadImage(dir_in)
        if write is True:
            sitk.WriteImage(sitk_image, dir_out)

        return sitk_image
