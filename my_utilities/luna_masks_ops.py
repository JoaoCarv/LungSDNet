# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    elipsoid(image_s, a_in, b_in, c_in, origin_in):
        builds an elipsoid mask for a lesion
    world2voxel(coord_w, origin):
        transforms the coordinates from the world refernce frame to voxels
    get_nodules_info(annotations_df,
                         names_convert_df,
                         filename_no_ext,
                         Origin,
                         nod_size_df):
        get information of lesions centroind and size for the given image name
"""
import numpy as np


def elipsoid(image_s, a_in, b_in, c_in, origin_in):
    """ Function for building a mask of an elipsoid around the lesion

    Parameters:
        image_s: (list of int) size of the image
        a_in: (int) size of the elipsoide in direction x
        b_in: (int) size of the elipsoide in direction y
        c_in: (int) size of the elipsoide in direction z
        origin_in: (list of ints) coordinates of the centroid of the lesion

    Returns:
        (int) number of elements
    """

    # Center of the Elipsoid -> need to change this according to max shift
    # (from Lunas16_ops) -> this will impact how z_shift is chosen
    origin = origin_in[::-1]
    x_0 = origin[0]
    y_0 = origin[1]
    z_0 = origin[2]

    a_0 = int(round(a_in/2))
    b_on = int(round(b_in/2))
    c_on = int(round(c_in/2))
    a_on = a_0

    # Image size
    image_size = image_s[::-1]
    image_size_x = image_size[0]
    image_size_y = image_size[1]
    image_size_z = image_size[2]

    # Define mask for elipsoid
    r = 1
    x, y, z = np.ogrid[-x_0:image_size_x-x_0,
                        -y_0:image_size_y-y_0,
                        -z_0:image_size_z-z_0]
    eq_l = x*x/(a_on*a_on) + y*y/(b_on*b_on) + z*z/(c_on*c_on)
    eq_r = r*r
    mask = eq_l <= eq_r

    image = np.zeros(image_size)
    image[mask] = 1
    # print(image)

    return image


def world2voxel(coord_w, origin):
    """ Function that converts the coordinates in the world reference frame to
    coordinates in voxels. THe spacing is hardcoded and assumes:
           [0.703000009059906,
           0.703000009059906,
           1.25]

    Parameters:
        coord: (list of int) coordinates in the world reference frame
        origin: (list of int) coordinates of the origin of the reference frame

    Returns:
        (int) coordinates of the origin
    """

    spacing = [0.703000009059906,
               0.703000009059906,
               1.25]

    # translation
    coord_t = np.absolute(coord_w[0:3] - origin[0:3])

    # stretching
    coord_v = coord_t/spacing

    return coord_v.astype(int)


def get_nodules_info(annotations_df,
                     names_convert_df,
                     filename_no_ext,
                     Origin,
                     nod_size_df):
    """ Function for getting the information of the lesion given its file name

    Parameters:
        annotations_df: (pandas DataFrame) info directly from LUNA
        names_convert_ff: (pandas DataFrame) correspondence between old name
                            and new name of the file
        filename_no_ext: (string) name of the file, without extension
        Origin: (list of ints) origin of the world reference frame
        nod_size_df: (pandas DataFrame) list of the lesions sizes

    Returns:
        out: (list) information of all the lesions in the image (origin in
                voxels and lesion size)
    """

    # origin
    name_seriesuid = names_convert_df.loc[names_convert_df['newname'] == filename_no_ext]
    name_seriesuid = name_seriesuid.iloc[0, 0]

    nodules_info = annotations_df.loc[annotations_df['seriesuid'] == name_seriesuid]
    number_of_nodules = int(nodules_info.size/5)

    nodules_origins_sizes = np.zeros((number_of_nodules, 4))
    for i in range(number_of_nodules):
        for j in range(4):  # [coord x, coord y, coord z, size]
            nodules_origins_sizes[i, j] = nodules_info.iloc[i, j+1]

    # nodules sizes in voxels
    nod_size = nod_size_df.loc[nod_size_df['name'] == filename_no_ext]
    # Prepare output structure
    out = []
    for i in range(number_of_nodules):

        org = nodules_origins_sizes[i, :]
        org_v = world2voxel(org, Origin)
        out2append = [org_v[0],
                      org_v[1],
                      org_v[2],
                      nod_size.iloc[i]['nodule size x (vx)'],
                      nod_size.iloc[i]['nodule size y (vx)'],
                      nod_size.iloc[i]['nodule size z (vx)']]

        out.append(out2append)

    return out
