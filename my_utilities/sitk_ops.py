"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    - nodules_connection(label_data, label_header):
        evaluates if the nodules are connected regions
    - bbox_coordinates(label_sitk):
        bounding box for a specific labeling image (1 object/im)
    - bbox_size(label_sitk)
        size of bounding box for a specific labeling image (1 object/im)
    - crop_bbox(img_sitk, label_sitk):
        determines the bounding box + gets data inside bbox
    - join_mask_lung(sitk_mask):
        assigns the same value to all the 3 labels in the mask
    - dilate_lung_mask(sitk_mask):
        corrects the lung masks, increasing the outline of the lung
    - remove_lungs(image_sitk, seg_sitk):
        applys the segmentation mask to the data
    - normalize_images(image_sitk):
        normalizes the voxel intensity [-1000,400]
"""

import SimpleITK as sitk
import numpy as np
from skimage import measure


def nodules_connection(label_data, label_header):

    """ Function evaluates if the nodules are connected regions, returns
    the different regions and evaluates the size of the nodules
    Libraries: numpy,skimage.measure

    Parameters:
        label_data:(Numpy array) image of the mask (0s & 1s) in array format
        label_header:(Nifty Header) header information regarding the nifty image

    Returns:
        las_label=(Numpy Array) array with the different regions defined with different numbers
        [xdif,ydiff,zdiff] = (list) lists the sizes of the nodules in the different directions
    """


    las_labels = measure.label(label_data,
                               neighbors=8,
                               background=0,
                               return_num=True)

    las_labels_nzero = np.nonzero(las_labels[0])
    [xdif, ydif, zdif] = [np.amax(las_labels_nzero[0])-np.amin(las_labels_nzero[0]),
                        np.amax(las_labels_nzero[1])-np.amin(las_labels_nzero[1]),
                        np.amax(las_labels_nzero[2])-np.amin(las_labels_nzero[2])]

    # conversion pixels to mm
    dims = label_header['pixdim']
    if label_header['xyzt_units'] == 10:
        #dimensions in mm
        print('xyzt_units=10')
        xdif=dims[1]*xdif
        ydif=dims[2]*ydif
        zdif=dims[3]*zdif


    return las_labels,[xdif,ydif,zdif]


def bbox_coordinates(label_sitk):

    """ Function determines the bounding box for a specific labeling image. Assumes
    one object per image

    Parameters:
        label_sitk:(Image SITK) labeled image of a nodule in sitk format

    Returns:
        bbox_pts: (Numpy Array matrix) Matrix 8x3 in which each row corresponds to a point of the bbox
    """

    #Setting Bounding Box
    F_statistics = sitk.LabelShapeStatisticsImageFilter()

    F_statistics.Execute(label_sitk)
    bbox_dims = F_statistics.GetBoundingBox(1)

    spacer = 3
    xmin = bbox_dims[0]-spacer
    xmax = bbox_dims[1]+spacer
    ymin = bbox_dims[2]-spacer
    ymax = bbox_dims[3]+spacer
    zmin = bbox_dims[4]-spacer
    zmax = bbox_dims[5]+spacer

    p1 = [xmin-spacer, ymin, zmin]
    p2 = [xmin, ymin, zmax]
    p3 = [xmin, ymax, zmin]
    p4 = [xmin, ymax, zmax]
    p5 = [xmax, ymin, zmin]
    p6 = [xmax, ymin, zmax]
    p7 = [xmax, ymax, zmin]
    p8 = [xmax, ymax, zmax]
    bbox_pts = [p1, p2, p3, p4, p5, p6, p7, p8]

    return bbox_pts

def bbox_size(label_sitk):
    """ Function determines the bounding box for a specific labeling image. Assumes
    one object per image. Only the size of the BBox is returned

    Parameters:
        label_sitk:(Image SITK) labeled image of a nodule in sitk format

    Returns:
        list: (Numpy Array) size 3 (size_x,size_y,size_z)
    """

    # Setting Bounding Box
    F_statistics = sitk.LabelShapeStatisticsImageFilter()

    F_statistics.Execute(label_sitk)
    bbox_dims = F_statistics.GetBoundingBox(1)
    return list(bbox_dims[3:6])


def crop_bbox(img_sitk, label_sitk):
    """ Function that determines the bounding box of the label, and also
    retrieves the data from the image, inside the assigned bbox.

    Parameters:
        img_sitk: (Image SITK) image converted through STIk
        label_sitk:(Image SITK) labeled image of a nodule in sitk format

    Returns:
        training_patch: (Image SITK) resulting image after application of
        the bounding box
    """

    # Setting Bounding Box
    F_statistics = sitk.LabelShapeStatisticsImageFilter()
    F_statistics.Execute(label_sitk)
    bbox_dims = F_statistics.GetBoundingBox(1)  # only one label per image
    # print(bbox_dims)

    # Applying the bounding box to the image with spacing equal to spc
    spc = 0
    org = bbox_dims[0:3] - [spc]*3
    sz = bbox_dims[3:6] + [spc]*3
    training_patch = img_sitk[org[0]-spc:org[0]+sz[0]+spc,
                              org[1]-spc:org[1]+sz[1]+spc,
                              org[2]-spc:org[2]+sz[2]+spc]

    return training_patch


def join_mask_lung(sitk_mask):
    """ Function that assigns the same value to all the 3 labels in each
    lung mask

    Parameters:
        sitk_mask: (SITK image object) original lung mask image

    Returns:
       sitk_mask_joint: (SITK image object) corrected lung mask

    """
    mask_nda = sitk.GetArrayFromImage(sitk_mask)
    mask_nda[mask_nda != 0] = 1

    sitk_mask_joint = sitk.GetImageFromArray(mask_nda)
    sitk_mask_joint.CopyInformation(sitk_mask)

    return sitk_mask_joint


def dilate_lung_mask(sitk_mask):
    """ Function that corrects the lung masks, order to have a larger
    outline of the lungs (don't lose lung)

    Parameters:
        sitk_mask: (SITK image object) original lung mask image

    Returns:
       sitk_mask_dilated: (SITK image object) corrected lung mask

    """

    filter = sitk.BinaryDilateImageFilter()
    filter.SetKernelRadius(7)
    sitk_mask_dilated = filter.Execute(sitk_mask, 0, 1, False)

    return sitk_mask_dilated


def remove_lungs(image_sitk, seg_sitk):
    """ Function applies the lung segmentation to the SITK image.

    Parameters:
        img_sitk: (Image SITK) image converted through STIk
        label_sitk:(Image SITK) labeled image of the lungs in sitk format

    Returns:
        image_no_lungs: (Image SITK) resulting image after application of the mask
    """

    # join labels in mask + dilate lungs
    seg_sitk_joint = join_mask_lung(seg_sitk)
    seg_sitk_dilate = dilate_lung_mask(seg_sitk_joint)

    # remove_lungs
    apply_mask = sitk.MaskNegatedImageFilter()
    background_value = 0
    image_no_lungs = apply_mask.Execute(image_sitk, seg_sitk_dilate, background_value, 1)
    # image_no_lungs = apply_mask.Execute(image_sitk, seg_sitk_joint, background_value, 1)

    return image_no_lungs


def normalize_images(image_sitk):
    """ Function that normalizes the image voxels intensity to the interval
    [-1000,400].

    Parameters:
        img_sitk: (Image SITK) image converted through STIk
    Returns:
        training_patch: (Image SITK) resulting image after the application of
            the cutoff
    """

    max = 400
    min = -1000

    image_np = sitk.GetArrayFromImage(image_sitk)

    # Normalization
    image_np = (image_np - min)/(max - min)
    image_np[image_np > 1] = 1
    image_np[image_np < 0] = 0

    # Convert back to SITK
    out_image_sitk = sitk.GetImageFromArray(image_np)
    out_image_sitk.CopyInformation(image_sitk)

    return out_image_sitk
