"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomuncações

Code:
    get_new_spacing(path_csv)
        finds the median spacing for each direction
    resample_image(in_image, new_spacing, interpolator)
        Ressamples an image with the interpolator to
        a new spacing defined by new_spacing

"""

import numpy as np
import pandas as pd
import SimpleITK as sitk

def get_new_spacing(csv_path_read):
    """
    From the information of a previously defined
    csv file with the spacing sizes, it defines
    the new spacing by using the median of all
    spacings
    csv_path_read : string
    returns: (ndarray) new spacing
    """

    spacing_stats = pd.read_csv(csv_path_read)

    # find median for every direction
    new_dims = []
    for i in range(3):
        new_dims.append(pd.DataFrame.median(spacing_stats.iloc[:, 2+i].round(3)))

    return np.array(new_dims)


def resample_image(in_image, new_spacing, interpolator):
    """
    Ressamples an image with the interpolator to
    a new spacing defined by new_spacing
    in_image : sitk image
    spacing : np.array
    interpolator : sitk' interpolators (
    https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5
    )

    """
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator = interpolator
    # resample.SetInterpolator = sitk.sitkNearestNeighbor
    resample.SetOutputDirection(in_image.GetDirection())
    resample.SetOutputPixelType = in_image.GetPixelIDValue()
    resample.SetOutputOrigin(in_image.GetOrigin())
    resample.SetOutputSpacing(new_spacing)

    # defining new size:
    orig_size = np.array(in_image.GetSize(), dtype=np.int)
    orig_spacing = np.array(list(in_image.GetSpacing()))
    new_spacing = np.array(new_spacing)
    new_size = orig_size*(orig_spacing/new_spacing)
    new_size = np.ceil(new_size).astype(np.int)
    new_size = [int(s) for s in new_size]

    resample.SetSize(new_size)

    new_image = resample.Execute(in_image)

    return new_image
