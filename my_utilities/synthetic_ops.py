# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomuncações

CODE:
    safe_elips_shift(window_size, a, b, safe_dist=5):
        define maximum shift of nodule according to elipse size
    shift(max_shift):
        define the shift of the nodule inside of the patch
    elipsoid_random_3d(Size, max_size, min_size):
        creates a 3D SITK image with a single elipsoid inside it.
    sizes_gamma_distribution(number_nodules, max_size,min_size):
        creates a list of values following a gamma distribution
    elipsoid_random_2d(Size, max_size, min_size, return_origin=False,
                       ab_given=False, a=0, b=0):
        creates a 2D SITK image with a single elipsoid inside it.
    give_noise(image, gaussian_noise=True, smoothing=True):
        creates a new image with smoothed gaussian noise
"""


import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter



def safe_elipse_shift(window_size, a, b, safe_dist=5):
    """ Function for defining the maximum shift of the nodules inside the
    patch, according to the size of the elipse.
    Parameters:
        window_size : (int) size of the windown (x=y)
        a : (int) size o elipse in x direction
        b : (int) size o elipse in y direction
        safe_dist : (int) maximum distance between the nodule walls and the
            patch frame

    Returns:
        (list of ints; size 2) the maximum shift in x and y direction
    """

    safe_shift_x = int(window_size/2) - a - safe_dist
    safe_shift_y = int(window_size/2) - b

    return [safe_shift_x, safe_shift_y]


def shift(max_shift):
    """ Function for defining the shift of the nodule inside of the patch

    Parameters:
        max_shift : (list of ints, size 2) maximum allowed shift

    Returns:
        (list of ints; size 2) shift of the nodule in voxels
    """

    shift_x = random.randint(-max_shift[0],
                             max_shift[0])
    shift_y = random.randint(-max_shift[1],
                             max_shift[1])

    return [shift_x, shift_y]


def elipsoid_random_3d(Size, max_size, min_size):

    """ Function creating an SITK image with a single elipsoid inside it.
    Disclaimer: inside we start with a numpy object and then transition todo
        a sitk object. It is import to take into account that the coordinates
        change order with that transformation. In the case of the square patch
        with random radius values, it has no impact. The same can't be always
        said other cases.

    Parameters:
        Size: (list of ints) Size of the image in each direction
        max_size: (int) maximum diameter of the patch
        min_size: (int) minimum diameter of the patch

    Returns:
        image : (numpy 3D tensor) image with a random nodule
    """

    # Center of the Elipsoid -> I will need to change this according to max
    # (from Lunas16_ops) -> this will impact how z_shift is chosen
    radius = int(Size/2)
    x_0 = radius
    y_0 = radius
    z_0 = radius
    max_size_2 = int(max_size/2)  # we want the maximum radius
    min_size_2 = int(min_size/2)

    # Size of the Elipsoid:
    a = random.randint(min_size_2, max_size_2)
    b = random.randint(min_size_2, max_size_2)
    c = random.randint(min_size_2, max_size_2)
    r = 1

    # apply shift to origin in x and z
    max_shift = safe_elipse_shift(window_size=Size,
                                  a=a,
                                  b=b)

    new_shift = shift(max_shift)
    x_0 = new_shift[0] + x_0
    y_0 = new_shift[1] + y_0

    # Corrections related to Z axis
    Size_z = int(c*2)  # redifining new z for the elipsoid
    z_shift = radius - c  # taking into account the shift in z

    # Define mask for elipsoid
    y, x, z = np.ogrid[-x_0:Size-x_0, -y_0:Size-y_0, -z_0:Size_z-z_0]
    z = z_shift*np.ones((1, 1, Size_z)) + z
    eq_l = x*x/(a*a) + y*y/(b*b) + z*z/(c*c)
    eq_r = r*r
    mask = eq_l <= eq_r

    image = np.zeros((Size, Size, Size_z))
    image[mask] = 1

    return image


def sizes_gamma_distribution(number_nodules, max_size, min_size):
    """Function which creates a list of values following a gamma distribution
    and that adapts it to fit the maximum size and minimum sized defined.

    Parameters:
        number_nodules: (int) size of the list
        max_size: (int) maximum diameter of the patch
        min_size: (int) minimum diameter of the patch

    Returns:
        s : (list of floats) list of values that follow the distribution, and
            are within the limits
    """
    k = 3
    theta = 2
    s = np.random.gamma(k, theta, number_nodules)
    s = s/max(s)
    s = s*max_size + min_size
    mean_size = np.mean(s)
    for i in range(s.size):
        if s[i] > max_size:
            s[i] = mean_size

    return s


def elipsoid_random_2d(Size, max_size, min_size, return_origin=False,
                       ab_given=False, a=0, b=0):

    """ Function that creates an SITK image with a single elipsoid inside it.
    Disclaimer: inside we start with a numpy object and then transition todo
        a sitk object. It is import to take into account that the coordinates
        change order with that transformation. In the case of the square patch
        with random radius values, it has no impact. The same can't be always
        said other cases.

    Parameters:
        Size: (list of ints) Size of the image in each direction
        max_size: (int) maximum diameter of the patch
        min_size: (int) minimum diameter of the patch
        return_origin (boolean) return the value of the origin used to define
                       center of the elipsoid
        ab_given (boolean) True means that the values for the elipsoid diameter
                           will be given


    Returns:
        image : (numpy matrix) 2D image with a random nodule
    """

    # Center of the Elipsoid -> I will need to change this according to max shift
    # (from Lunas16_ops) -> this will impact how z_shift is chosen
    radius = int(Size/2)
    x_0 = radius
    y_0 = radius
    max_size_2 = int(max_size/2)  # we want the maximum radius
    min_size_2 = int(min_size/2)

    # Size of the Elipsoid:
    if ab_given is False:
        a = random.randint(min_size_2, max_size_2)
        b = random.randint(min_size_2, max_size_2)
    else:
        b = round(a/2 + (a/2)*random.uniform(0, 0.1))
        a = round(a/2)
        if random.randint(0, 1) == 1:
            b, a = a, b
    r = 1

    # apply shift to origin in x and z
    max_shift = safe_elipse_shift(window_size=Size,
                                  a=a,
                                  b=b)

    new_shift = shift(max_shift)
    x_0 = new_shift[0] + x_0
    y_0 = new_shift[1] + y_0

    # Define mask for elipsoid
    y, x = np.ogrid[-x_0:Size-x_0, -y_0:Size-y_0]
    eq_l = x*x/(a*a) + y*y/(b*b)
    eq_r = r*r
    mask = eq_l <= eq_r

    image = np.zeros((Size, Size))
    image[mask] = 1
    # print(image)
    if return_origin:
        return image, x_0, y_0, a, b
    else:
        return image


def give_noise(image, gaussian_noise=True, smoothing=True):
    """ Function that adds gaussian noise following a distribution with
    mean = 0.6 and stf = 0.1. This is followed by gaussian filtering with
    a kernel with std = 2.

    Parameters:
        image: (numpy tensor) image to be filtered/added noise to

    Returns:
        image : (numpy tensor) image with noise
    """

    if gaussian_noise:
        # applying random gaussian noise
        if len(image.shape) == 2:
            row, col = image.shape
            mean = 0.6
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)

        else:
            row, col, ch = image.shape
            mean = 0.6
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
        image = image + gauss

    if smoothing:
        # apply gaussian kernel (smothness)
        image = gaussian_filter(input=image, sigma=1)
        image[image < 0.1] = 0

    return image
