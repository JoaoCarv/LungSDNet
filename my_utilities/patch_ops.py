"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    global2patch(coord, window_width):
        converts coordinates from global coord to the patch coordinates
        according to the size of the patch
    random_patch(windwo_widt,size):
        defines a set of 3 coordinates to be the origin of the new patch
    max_shift(window_width, width, safe_distance, org_v, Size):
        defines max shift appliable to the patch
    secure_inside_full_window(shift,Size,Safe_Distance,sz2,org_v):
        verifies if a patch is inside a image full width
    far2nodules(nodules_origins_sizes, rand_o):
        verifies if a random patch doens't contain nodules
    close2lungs(original_sitk_image, org_v):
        verifies if random patch is close the lungs
    shift2original(max_shift_in):
        defines the shift in the original image coordinates
    mm2voxel(spacing, mm):
        converts from mm to voxel
    world2voxel(coord_w, origin, spacing):
        convert from world coordinates to voxel coordinates in the images
"""

import random
import numpy as np


def global2patch(coord, window_width):
    """ Function converts coordinates from the full image size to the
    patch coordinates according to the size of the patch.

    Parameters:
        coord: (list of 3 floats) coordinates to be converted
        window_width: (int) number of voxels that define the window width

    Returns:
        (list of ints, size 2) coordinates in the patch size frame
    """
    return [int(window_width / 2), int(window_width / 2)]


def random_patch(window_width, size):
    """ Function defines a set of 3 coordinates to be the origin of the
    new patch.

    Parameters:
        window_width: (int) number of voxels that define the window width
        size: (list of ints) size in voxels of the patch

    Returns:
        (list of int, size 3) coordinates of the origin for the new patch
    """
    width_secure = int(window_width / 2)
    x = random.uniform(1 + width_secure, size[0] - 2 - width_secure)
    y = random.uniform(1 + width_secure, size[1] - 2 - width_secure)
    z = random.uniform(0, size[2] - 2)

    return [int(x), int(y), int(z)]


def max_shift(window_width, width, safe_distance, org_v, Size):
    """ Function that defines the max shift that can be applied to the patch
    without it leaving the image frame

    Parameters:
        window_width: (int) number of voxels that define the window width
        width : (int) number of voxels that comprise the diameter of the nodule
        safe_distance: (int) number of voxels to leave around the patch
        org_v: (int) origin of the nodule in voxels
        Size: (list of ints) Size of the image in each direction

    Returns:
        max_shift_n: (list of ints, size 4) maximum shift
            for x and y in both ways (-x_i and +x_i)

    """

    half_wind = int(window_width / 2)
    half_diam = int(width / 2)  # radius of the nodule segmentation
    max_shift_n = half_wind - half_diam - safe_distance
    while half_wind - max_shift_n <= safe_distance or max_shift_n < 0:
        max_shift_n += 1

    max_shift_n = [max_shift_n] * 4
    # guards for shifted window outside image
    if max_shift_n[0] + half_wind > org_v[0]:
        max_shift_n[0] = org_v[0] - safe_distance - half_wind

    if max_shift_n[1] + half_wind > Size[0] - org_v[0]:
        max_shift_n[1] = Size[0] - org_v[0] - safe_distance - half_wind

    if max_shift_n[2] + half_wind > org_v[1]:
        max_shift_n[2] = org_v[1] - safe_distance - half_wind

    if max_shift_n[3] + half_wind > Size[1] - org_v[1]:
        max_shift_n[3] = Size[1] - org_v[1] - safe_distance - half_wind

    return max_shift_n


def secure_inside_full_window(shift, Size, Safe_Distance, sz2, org_v):
    """ Function that verifies if a patch is inside a image full width

    Parameters:
        window_width: (int) number of voxels that define the window width
        Size: (list of ints) Size of the image in each direction
        Safe_Distance: (int) number of voxels to leave around the patch
        sz2 : (int) half size of the patch window
        org_v: (int) origin of the nodule in voxels


    Returns:
        inside: (boolean) True if inside, False if outside
    """

    inside = True

    # confirm np.array
    shift = np.array(shift)
    patch_window_size = np.array([sz2] * 2)
    origin_nodule = np.array(org_v[0:2])
    final_im_org = origin_nodule - patch_window_size + shift
    if final_im_org[0] < 0 + Safe_Distance:
        inside = False
    elif final_im_org[1] < 0 + Safe_Distance:
        inside = False
    elif final_im_org[0] > Size[0] - Safe_Distance:
        inside = False
    elif final_im_org[1] > Size[1] - Safe_Distance:
        inside = False

    return inside


def far2nodules(nodules_origins_sizes,
                rand_o,
                nodules_size_z,
                Origin,
                Spacing,
                context=False):
    """ Function that verifies if random patch is close to a nodule

    Parameters:
        nodules_origins_sizes: (ndarray) array with locations of nodules
        rand_o: (list of ints) random origin (coordinates x and y)
        nodules_size_z: (list) sizes of nodules in the direction z
        Origin: (list) origin of the image
        Spacing: (list) Spacing of the image
        context: (boolean) if True, the dgr_z in z direction takes into
                    account the 3 slices for image context

    Returns:
        far: (boolean) False if has nodule, True if doesn't have nodule
    """
    far = True
    for org, size_z in zip(nodules_origins_sizes, nodules_size_z):
        org_v = world2voxel(org, Origin, Spacing)

        # check close to slice (z direction)
        size_z = int(size_z/2)
        if context is True:
            size_z += 1

        dgr_z = [org_v[2]-size_z, org_v[2]+size_z]

        no_nodule_z = (
            (rand_o[2] > dgr_z[0])
            and (rand_o[2] < dgr_z[1])
            )

        if no_nodule_z is True:
            return False
            break

        else:  # check close to nodule within slice
            dgr_z = [org_v[0] - 64,
                     org_v[0] + 64,
                     org_v[1] - 64,
                     org_v[1] + 64]

            no_nodule_cond = (
                (rand_o[0] > dgr_z[0])
                and (rand_o[0] < dgr_z[1])
                and (rand_o[1] > dgr_z[2])
                and (rand_o[1] < dgr_z[3])
            )

            if no_nodule_cond is False:
                far = False
                break

    return far


def close2lungs(sitk_image_only_lungs, org_v):
    """ Function that verifies if random patch is close the lungs

    Parameters:
        original_sitk_image: (SITK image object) original image
        org_v: (int) origin of the nodule in voxels


    Returns:
        close: (boolean) True if inside, False if outside
    """

    close = True
    background = 0
    if sitk_image_only_lungs[org_v[0], org_v[1], org_v[2]] == background:
        if random.uniform(0, 10) > 2:
            close = False

    return close


def shift2original(max_shift_in):
    """ Function that definies the shift in original image coordinate frame

    Parameters:
        max_shift_in: (list ints, size 4) max shift allowed in both ways of
            each coordinate.

    Returns:
       (list ints, size 2) shift in each direction

    """

    shift_x = random.uniform(-max_shift_in[0], max_shift_in[1])
    shift_y = random.uniform(-max_shift_in[2], max_shift_in[3])
    return [int(shift_x), int(shift_y)]


def mm2voxel(spacing, mm):
    """ Function converts mm to voxels according to the spacing of being used

    Parameters:
        spacing: (list of 3 ints) Spacing in each direction
    Returns:
        mm: (list of 3 floats) value of in mm
    """
    return mm / spacing


def world2voxel(coord_w, origin, spacing):
    """ Function converts from world coordinates to the system of
    coordinates of the image, according to its origin and Spacing
    in each direction

    Parameters:
        coord_w: (list of 3 floats) coordinates to be converted
        origin: (list of 3 ints) coordinates to origin in the image
        spacing: (list of 3 ints) Spacing in each direction of the image
    Returns:
        coord_v: (list of 3 ints) coordinates in voxels
    """

    # translation
    coord_t = np.absolute(coord_w[0:3] - origin[0:3])

    # stretching
    coord_v = coord_t / spacing

    return coord_v.astype(int)
