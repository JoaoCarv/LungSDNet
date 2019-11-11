"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    LIDC-IDRI FUNCTIONS:
    luna_crate_wnod_patch(org_v,N_patches,max_shift,image,sz2,k)
        extract a 2D patch (1 slice), with a lesion, from the image
    luna_crate_wnod_patch_context(org_v,N_patches,max_shift,image,sz2,k)
        extract a 2.5D patch (3 slice), with a lesion, from the image
    luna_create_random_patch(org_v,N_patches,max_shift,image,sz2,k):
        extracts a 2D random patch from the image (no lesion)
    luna_create_random_patch_context(nodules_origins_sizes,
            window_width,Size,image,nodules_size_z,Origin,Spacing,sz2):
        extracts a 2D random patch from the image (no lesion)
    DECATHLON FUNCTIONS:
    deca_create_3d_patches(bbox_dims,rs_size_z,Max_size,sizes_n,
            sitk_image,sitk_mask)
        extract a 3D patch from the image, centered around the lesion

"""

import math
import numpy as np
import patch_ops


def luna_create_wnod_patch(org_v,
                           N_patches,
                           max_shift,
                           image,
                           sz2,
                           k):
    """ Function that given an image and the coordinates of the lesion's
    centroid, outputs the patch (2D) containing the lesion.


    Parameters:
        org_v: (list) coordinates of the lesion's centroid
        N_patches: (int) number of patches to be exctracted from the image
        max_shift: (int) max shift that to the image so that the lesion isn't
                    in the center of the patch
        image: (sitk image) CT scan
        sz2: (int)  size of the patch in voxels

    Returns:
        patch: (sitk image) 2D patch extracted from the image
    """

    org_v_z = org_v[2] - int(N_patches / 2) + k
    shift = patch_ops.shift2original(max_shift)
    patch = image[
        org_v[0] - sz2 + shift[0]: org_v[0] + sz2 + shift[0],
        org_v[1] - sz2 + shift[1]: org_v[1] + sz2 + shift[1],
        org_v_z: org_v_z + 1,
    ]
    return patch


def luna_create_wnod_patch_context(org_v,
                                   N_patches,
                                   max_shift,
                                   image,
                                   sz2,
                                   k):
    """ Function that given an image and the coordinates of the lesion's
    centroid, outputs the patch containing the lesion, with one slice below
    and one below.


    Parameters:
        org_v: (list) coordinates of the lesion's centroid
        N_patches: (int) number of patches to be exctracted from the image
        max_shift: (int) max shift that to the image so that the lesion isn't
                    in the center of the patch
        image: (sitk image) CT scan
        sz2: (int)  size of the patch in voxels

    Returns:
        patch_minus: (sitk image) patch extracted from the image, below the
                        the main slice
        patch_base: (sitk image) patch extracted from the image (main slice)
        patch_plus: (sitk image) patch extracted from the image, above the
                        the main slice
    """

    org_v_z = org_v[2] - int(N_patches / 2) + k
    shift = patch_ops.shift2original(max_shift)
    patch_minus = image[
                  org_v[0] - sz2 + shift[0]: org_v[0] + sz2 + shift[0],
                  org_v[1] - sz2 + shift[1]: org_v[1] + sz2 + shift[1],
                  org_v_z: org_v_z + 1,
                  ]

    patch_base = image[
                  org_v[0] - sz2 + shift[0]: org_v[0] + sz2 + shift[0],
                  org_v[1] - sz2 + shift[1]: org_v[1] + sz2 + shift[1],
                  org_v_z + 1: org_v_z + 2,
                  ]

    patch_plus = image[
                  org_v[0] - sz2 + shift[0]: org_v[0] + sz2 + shift[0],
                  org_v[1] - sz2 + shift[1]: org_v[1] + sz2 + shift[1],
                  org_v_z + 2: org_v_z + 3,
                  ]

    return patch_minus, patch_base, patch_plus


def luna_create_random_patch(nodules_origins_sizes,
                             window_width,
                             Size,
                             image,
                             nodules_size_z,
                             Origin,
                             Spacing,
                             sz2):
    """ Function creates a random patch that is inside/close to the lungs
    and that doesn't contain any other nodule

    Parameters:
        nodules_origins_sizes: (ndarray) array with locations of nodules
        window_width: (int) number of voxels that define the window width
        size: (list of ints) size in voxels of the patch
        image: (SITK image object) image with only the nodules (background=0)
        nodules_sizes_z: (pandas dataframe) contains the nodules sizes in the z
                direction, for all the nodules in the image being prepared
        Origin: (list) Origin of the image
        Spacing: (list) Spacing of the image
        sz2: (int)  size of the patch in voxels

    Returns:
        rand_patch: (SITK image object) 2D random patch
        rand_o: (ndarray) coordinates of the random patch
    """

    b = False
    while b is False:  # Find a random origin outside the D zone

        # Get patch origin
        rand_o = patch_ops.random_patch(window_width, Size)

        # Evaluate if outside danger zone (outside lungs + close to nodule)
        cond1 = patch_ops.close2lungs(image, rand_o)
        if cond1 is True:
            cond2 = patch_ops.far2nodules(nodules_origins_sizes,
                                          rand_o,
                                          nodules_size_z,
                                          Origin,
                                          Spacing)
            if cond2 is True:
                b = True

    rand_patch = image[
        rand_o[0] - sz2: rand_o[0] + sz2,
        rand_o[1] - sz2: rand_o[1] + sz2,
        rand_o[2]: rand_o[2] + 1,
    ]

    return rand_patch, rand_o


def luna_create_random_patch_context(nodules_origins_sizes,
                                     window_width,
                                     Size,
                                     image,
                                     nodules_size_z,
                                     Origin,
                                     Spacing,
                                     sz2):
    """ Function creates a random 2.5D patch that is inside/close to the lungs
    and that doesn't contain any other nodule

    Parameters:
        nodules_origins_sizes: (ndarray) array with locations of nodules
        window_width: (int) number of voxels that define the window width
        size: (list of ints) size in voxels of the patch
        image: (SITK image object) image with only the nodules (background=0)
        nodules_sizes_z: (pandas dataframe) contains the nodules sizes in the z
                direction, for all the nodules in the image being prepared

    Returns:
        rand_patch: (SITK image object) 2D random patch
        patch_minus: (SITK image object) random patch from the image, below the
                        the main slice
        patch_base: (SITK image object)random patch from the image (main slice)
        patch_plus: (SITK image object) random patch  from the image, above the
                        the main slice
    """

    b = False
    while b is False:  # Find a random origin outside the D zone

        # Get patch origin
        rand_o = patch_ops.random_patch(window_width, Size)

        # Evaluate if outside danger zone (outside lungs + close to nodule)
        cond1 = patch_ops.close2lungs(image, rand_o)
        if cond1 is True:
            cond2 = patch_ops.far2nodules(nodules_origins_sizes,
                                          rand_o,
                                          nodules_size_z,
                                          Origin,
                                          Spacing,
                                          context=True)
            if cond2 is True:
                b = True

    rand_patch_minus = image[
        rand_o[0] - sz2: rand_o[0] + sz2,
        rand_o[1] - sz2: rand_o[1] + sz2,
        rand_o[2]-1: rand_o[2],
    ]
    rand_patch_base = image[
        rand_o[0] - sz2: rand_o[0] + sz2,
        rand_o[1] - sz2: rand_o[1] + sz2,
        rand_o[2]: rand_o[2] + 1,
    ]
    rand_patch_plus = image[
        rand_o[0] - sz2: rand_o[0] + sz2,
        rand_o[1] - sz2: rand_o[1] + sz2,
        rand_o[2]+1: rand_o[2] + 2,
    ]

    return rand_patch_minus, rand_patch_base, rand_patch_plus, rand_o


def deca_create_3d_patches(bbox_dims,
                           rs_size_z,
                           Max_size,
                           sizes_n,
                           sitk_image,
                           sitk_mask):
    """ Function exctracts a 3D patch around the lesion, containing 50% more
    slices distibuted below and above the lesion

    Parameters:
        bbox_dims: (ndarray) array with dimension of bounding box around lesion
        rs_size_z: (pandas dataframe) sizes of the lesions after resampling
        Max_size: (list of ints) size of the window
        sitk_image: (SITK image object) image with only the nodules
        sitk_mask: (SITK image object) mask of the image  segmnetation

    Returns:
        patch_image: (SITK image object) 3D patch exctrated from image
        patch_mask: (SITK image object) 3D patch extracted from maks of the
                    segmentation
        patch_base: (SITK image object)random patch from the image (main slice)
        patch_plus: (SITK image object) random patch  from the image, above the
                        the main slice
    """

    # Applying the bounding box to the image with spacing equal to spc
    org = np.array(list(bbox_dims[0:3]))
    sz = np.array(list(bbox_dims[3:6]))
    sz_2_for_ORG = sz/np.array([2, 2, 2])
    sz_2_for_ORG = sz_2_for_ORG.astype(int)

    ORG = org + sz_2_for_ORG
d
    # sz2 for resampled images
    sz1 = int(Max_size[0]/2)
    sz2 = int(math.ceil(rs_size_z.iloc[sizes_n]/2))
    sz2 = sz2 + int(sz2/2)
    sizes_n += 1

    # print('size :',list(sitk_image.GetSize()))
    # print('origin: ', ORG)

    patch_image = sitk_image[ORG[0] - sz1:ORG[0] + sz1,
                             ORG[1] - sz1:ORG[1] + sz1,
                             ORG[2] - sz2:ORG[2] + sz2]
    patch_mask = sitk_mask[ORG[0] - sz1:ORG[0] + sz1,
                           ORG[1] - sz1:ORG[1] + sz1,
                           ORG[2] - sz2:ORG[2] + sz2]

    return patch_image, patch_mask
