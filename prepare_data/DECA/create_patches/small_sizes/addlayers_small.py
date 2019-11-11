# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

Script that for each image in the dataset adds creates two new images that
are shifted with regard to the transverse axis. "..._plus.nii.gz" is
positively shifted, and "..._minus.nii.gz" is negatively shifted.

Required changes are:
- path of the directory of the images (DirectoryOfDataset)
"""

import os
import nibabel as nib
import numpy as np
from shutil import copyfile
import SimpleITK as sitk

directoryOfDataset = r''
print(directoryOfDataset)
outputDir = os.path.join(directoryOfDataset, "context")
directoryOfImages = os.path.join(directoryOfDataset, "images")

for file in os.listdir(directoryOfImages):  # runs through all the image files

    filename = os.fsdecode(file)
    if filename.endswith('.nii.gz') and '._' not in filename:
        dirofpatient = os.path.join(directoryOfImages, filename)
        # dirofpatient = directoryOfImages + "/" + filename
        print(dirofpatient)

        img = nib.load(dirofpatient)

        headerimg = img.header

        Out = img.get_fdata()
        a = np.zeros(Out.shape[0:2]+(1,))

        outPlus1 = np.append(Out, a, axis=2)

        # eliminar a primeira camada:
        outPlus1 = np.delete(outPlus1, 0, 2)

        outMinus1 = np.append(a, Out, axis=2)
        # eliminar a ultima camada:
        outMinus1 = np.delete(outMinus1, outMinus1.shape[2]-1, 2)

        originalaffine = img.affine

        new_image_plus = nib.Nifti1Image(outPlus1, affine=originalaffine)
        new_image_minus = nib.Nifti1Image(outMinus1, affine=originalaffine)

        endposition = filename.find('.nii.gz')
        patient = filename[0:endposition]

        nib.save(img, os.path.join(outputDir, patient+'base.nii.gz'))
        nib.save(new_image_plus, os.path.join(outputDir, patient+'plus.nii.gz'))
        nib.save(new_image_minus, os.path.join(outputDir, patient + 'minus.nii.gz'))
        print(patient + '   done')
