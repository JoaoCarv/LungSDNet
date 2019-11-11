import sys
import os
import SimpleITK as sitk
module_path = r'..\my_utilities'
sys.path.insert(0, module_path)
import sitk_ops as ops

path_dir = r'C:\JFCImportantes\Universidade\Thesis\dataset\LUNA16\test\lung-seg'
path_dir_images = r'C:\JFCImportantes\Universidade\Thesis\dataset\LUNA16\test\images'
image_name = 'image_0_001.nii.gz'
path_image_lung = os.path.join(path_dir, image_name)
path_image = os.path.join(path_dir_images, image_name)
sitk_mask = sitk.ReadImage(path_image_lung)
sitk_image = sitk.ReadImage(path_image)


sitk_joint_mask = ops.join_mask_lung(sitk_mask)
sitk_out_mask = ops.dilate_lung_mask(sitk_joint_mask)
sitk_no_lungs = ops.remove_lungs(sitk_image, sitk_out_mask)
path_out_image = os.path.join(path_dir, 'dil2_' + image_name)
path_out_image_noluns = os.path.join(path_dir_images, 'no_lung2_'+image_name)


# sitk.WriteImage(sitk_out_mask, path_out_image)
sitk.WriteImage(sitk_no_lungs, path_out_image_noluns)
