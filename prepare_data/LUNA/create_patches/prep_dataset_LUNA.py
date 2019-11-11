# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

Script will apply all the preprocessing steps to the image followed by
the weighted sampling of patches from it.

Required chages are:
- path of the directory of the ressampled images (Directory_resampled)
"""

import os
import SimpleITK as sitk
import csv
import pandas as pd
import numpy as np

import sys

module_path = r"..\..\..\..\my_utilities"
sys.path.insert(0, module_path)
import utilities as utl
import sitk_ops
import patch_ops
import create_patches

# Parameters
Removed_Lungs_Save = False

# ---------- Directory informations
# Define directorys
# Directory_raw = r'C:\JFCImportantes\Universidade\Thesis\dataset\LUNA16\test'
Directory_resampled = r""
dir_images = os.path.join(Directory_resampled, "images_wnod")
# out directorys
dir_out = os.path.join(Directory_resampled, "Nvar_regress_class")
dir_out_no_lungs = os.path.join(
    dir_out, "no-lungs"
)  # only if removed_lungs_Save is True
dir_out_patches = os.path.join(dir_out, "patches")
dir_out_patches_regress = os.path.join(dir_out, "patches_regress")

if os.path.exists(dir_out_patches_regress) is False:
    os.mkdir(dir_out_patches_regress)

if os.path.exists(dir_out_no_lungs) is False:
    os.mkdir(dir_out_no_lungs)

dir_segment = os.path.join(Directory_resampled, "segs")

# ----------- CSV files
# in csvs
annotations_path = os.path.join(Directory_resampled, "annotations.csv")
annotations_df = pd.read_csv(annotations_path)

names_convert_path = os.path.join(Directory_resampled, "name_convert.csv")
names_convert_df = pd.read_csv(names_convert_path)

new_sizes_csv_path = os.path.join(Directory_resampled, "nodules_sizes.csv")
new_sizes_df = pd.read_csv(new_sizes_csv_path)
df_new_sizes_size_z = new_sizes_df["nodule size z (vx)"]
df_new_sizes_size_mm = new_sizes_df["nodule_size (mm)"]

# out csvs
path_failed_annotations = os.path.join(dir_out, "failed_annotations.csv")
path_origins_voxels = os.path.join(dir_out, "origins_voxels_Nvar.csv")
# names_convert = pd.read_csv(path_names)
path_patch_csv = os.path.join(dir_out, "nodules_class.csv")
path_patch_regress_csv = os.path.join(dir_out, "nodules_regress.csv")


# Run through all resampled images
images_numb = 1
failed_count = 0
nodule_number = 0


for file in os.listdir(dir_images):

    filename = os.fsdecode(file)
    if filename.endswith(".nii.gz") and filename[0] != ".":

        # ---------- Import images
        filename_no_ext = os.path.splitext(filename[:-3])[0]

        print(images_numb, '|', filename)
        images_numb += 1
        path_image = os.path.join(dir_images, filename)
        path_seg = os.path.join(dir_segment, filename)

        # Read image
        image_sitk = sitk.ReadImage(path_image)
        seg_sitk = sitk.ReadImage(path_seg)

        Origin = np.array(list(image_sitk.GetOrigin()))
        Spacing = np.array(list(image_sitk.GetSpacing()))
        Size = np.array(list(image_sitk.GetSize()))

        # ----------Remove Lungs
        image = sitk_ops.remove_lungs(image_sitk, seg_sitk)

        # Save Lungs
        if Removed_Lungs_Save is True:
            path_no_lungs_img = os.path.join(dir_out_no_lungs, filename)
            sitk.WriteImage(image, path_no_lungs_img)

        # ---------- Get Nodules
        # Get information about bboxex
        name_seriesuid = names_convert_df.loc[
            names_convert_df["newname"] == filename_no_ext
        ]
        name_seriesuid = name_seriesuid.iloc[0, 0]


        nodules_info = annotations_df.loc[annotations_df["seriesuid"] == name_seriesuid]

        number_of_nodules = int(nodules_info.size / 5)
        # Evaluate if the name is is in annotations
        if number_of_nodules < 1:
            row = [name_seriesuid, filename_no_ext]
            with open(path_failed_annotations, "a") as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            failed_count += 1

        # Transform coordinates to voxel frame
        # Get nodules locations and sizes
        nodules_origins_sizes = np.zeros((number_of_nodules, 4))
        for i in range(number_of_nodules):
            for j in range(4):  # [coord x, coord y, coord z, size]
                nodules_origins_sizes[i, j] = nodules_info.iloc[i, j + 1]

        # Define new window
        max_nodule_size = 39  # voxels (from max_size.py)
        window_width = 64
        sz2 = int(window_width / 2)

        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        i = 0
        r = 0
        N_rand = 0
        # ------------- Creating the Patches from the image
        for org in nodules_origins_sizes:
            s = 0
            # Convert to voxel coordinates
            org_v = patch_ops.world2voxel(org, Origin, Spacing)

            # Define Safe Distance for random shitf in original nodule patch
            Safe_Distance = 3
            # define bounding box
            width = [org[3]] * 3 / Spacing

            max_shift = patch_ops.max_shift(
                window_width, width[0], Safe_Distance, org_v, Size
            )

            # Number of patches according to size of nodule
            N_patches = int(df_new_sizes_size_z.iloc[nodule_number] / 2)
            nodule_number += 1

            # Define N_patches patches from each nodule
            for k in range(N_patches):
                N_rand += 1
                org_v_z = org_v[2] - int(N_patches / 2) + k
                shift = patch_ops.shift2original(max_shift)
                # ------------- From Nodule
                patch = image[
                    org_v[0] - sz2 + shift[0]: org_v[0] + sz2 + shift[0],
                    org_v[1] - sz2 + shift[1]: org_v[1] + sz2 + shift[1],
                    org_v_z: org_v_z + 1,
                ]
                patch_name = (
                    filename_no_ext + "_" + alphabet[r] + str(s + 1) + ".nii.gz"
                )
                patch_path = os.path.join(dir_out_patches, patch_name)
                patch_path_regress = os.path.join(dir_out_patches_regress,
                                                  patch_name)

                # Normalize and reduce intensity windown
                # patch = lops.normalize_images(patch)

                # Write Patch
                sitk.WriteImage(patch, patch_path)
                sitk.WriteImage(patch, patch_path_regress)

                # append info to csv file groundtruth
                row = [patch_name[6:-7], 1]
                row.append(df_new_sizes_size_mm.iloc[nodule_number-1])
                row.append(0)
                row.append(0)
                row.append(0)

                with open(path_patch_csv, "a") as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()
                with open(path_patch_regress_csv, "a") as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()

                # append info to origins csv
                row = [patch_name[0:-7], org_v[0], org_v[1]]

                with open(path_origins_voxels, "a") as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()

                i += 1
                s += 1

            r += 1

        # if number_of_nodules < 1:
        #     N_rand = 10
        N_rand = 20

        # ------------- Random Patches
        for k in range(N_rand):

            nodules_size_z = new_sizes_df.loc[
                        new_sizes_df['name'] == filename_no_ext
            ]["nodule size z (vx)"]
            patch, rand_o = create_patches.luna_create_random_patch(nodules_origins_sizes,
                                                                    window_width,
                                                                    Size,
                                                                    image,
                                                                    nodules_size_z,
                                                                    Origin,
                                                                    Spacing,
                                                                    sz2)

            patch_name = (
                filename_no_ext + "_rand" + str(k + 1) + ".nii.gz"
            )

            patch_path = os.path.join(dir_out_patches, patch_name)

            # Normalize and reduce intensity windown
            # patch = lops.normalize_images(patch)

            # Write Patch
            sitk.WriteImage(patch, patch_path)

            # Save info to csv of groundtruth
            row = [patch_name[6:-7], 0, 0, 0, 0, 0]

            with open(path_patch_csv, "a") as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()

            # append info to origins csv
            row = [patch_name[6:-7], rand_o[0], rand_o[1]]

            with open(path_origins_voxels, "a") as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()



    utl.beep_sound()
