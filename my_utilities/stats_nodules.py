# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomuncações

CODE:
    stats_nodules_size(s_nodules_sizes):
        evaluates the statistics of the nodules sizes (given the csv file)
    metrics_nodules: runs through the full decathlon dataset and gives its
        statistics (volume, percentage volume, largest and Feret diameter)
    stats_nodules_deca: calculates the extremes, mean and std of several metrics
        given by the values_dic

"""

import os
import utilities as utl
import numpy as np
import csv
from skimage import measure
import pandas as pd
import SimpleITK as sitk
from pandas import DataFrame as df


def stats_nodules_size(s_nodules_sizes):
    """ Function that evaluates the statistics of the nodules sizes


    Parameters:
        s_nodules: (pandas Series) series with all the nodules sizes


    Returns:
        values_dic (dictionary):
            max - max nodule size
            min - min nodule size
            std - standard deviation of the nodules sizes
            mean - mean of the nodules sizes
    """
    values_dic = {}
    values_dic['max'] = s_nodules_sizes.max()
    values_dic['min'] = s_nodules_sizes.min()
    values_dic['std'] = df.std(s_nodules_sizes)
    values_dic['mean'] = df.mean(s_nodules_sizes)

    return values_dic



def metrics_nodules():
    """ Function runs through all the images in the dataset (using the file
    patientsnumb.txt), and evaluates the volume, percentage volume, largest
    diameter and Feret diameter of each nodule. It saves this information in
    csv file (stats_nodules.csv) and also returns it in a dictionary

    Parameters:


    Returns:
        values_dic (dictionary):
            volume - list with the volumes of the nodules
            volume% - list with the percentage volumes of the nodules
            larg_diam - list with the largest diameters of the nodules
            Feret_diam - list with the feret diameters of the nodules
    """


    g=open(r"C:\JFCImportantes\Universidade\Thesis\dataset\images\patientsnumb.txt")
#    dimtxt=open("metrics_nodules.txt","w+") #file
#    dimtxt.write("number"+" | " + "volume" + " | " + "largest diam" +" | " + "perpend diam" +"\r\n")

    #Creates a .txt file with the patients with more than 1 nodule
    b=False
    with open('stats_nodules.csv', mode='w') as stats_file:
        fieldnames = ['number', 'volume', 'diam','diam2', 'diam3','Feret diam']
        writer = csv.DictWriter(stats_file, delimiter=';',fieldnames=fieldnames)
        writer.writeheader()

        #----Statistics
        values_dic = {'volume':[],
                      'volume%':[],
                      'larg_diam':[],
                      'Feret diam':[]}

        while 1:

            number=g.readline(3)

            if number=="096":
                b=True

            #ignore lines without numbers
            if number!="\n":  #ignore lines without numbers
                print(number)

                #----------Prep
                a=niftiloader(number,[0,1,0,1,0,1])
                dims=a.get(2).header['pixdim']
                label_data=a.get(4)
                voxel_volume=dims[1]*dims[2]*dims[3]
                img_size=np.size(label_data)

                #----------Calc Metrics
                #calc volume
                las_labels = measure.label(label_data,neighbors=8, background=0,return_num=True)
                las_labels_nzero=np.count_nonzero(las_labels[0])
                volume=las_labels_nzero*voxel_volume

                # Feret Diameter + Oriented Diameters
                F_statistics = sitk.LabelShapeStatisticsImageFilter()
                F_statistics.ComputeFeretDiameterOn()
                F_statistics.ComputeOrientedBoundingBoxOn()

                F_statistics.Execute(a.get(6))

                diam = F_statistics.GetFeretDiameter(1)
                obb = F_statistics.GetOrientedBoundingBoxSize(1)

                # -------Save to dict stats
                values_dic['volume'].append(volume)
                values_dic['volume%'].append(las_labels_nzero/img_size*100)
                values_dic['larg_diam'].append(max(obb))
                values_dic['Feret diam'].append(diam)

                # -------Write to csv
                writer.writerow({'number' : number,
                                 'volume' : str(volume),
                                 'diam' : str(obb[0]),
                                 'diam2' : str(obb[1]),
                                 'diam3' : str(obb[2]),
                                 'Feret diam' : str(diam)})

            if not number or b:
                break

    # ------- Clean up and finish
    g.close()

    utl.beep_sound()

    return values_dic


def stats_nodules_deca(values_dic):
    """ Function that calculates the extremes, mean and std of several metrics
    given by the values_dic

    Parameters:
        values_dic (dictionary):
            volume - list with the volumes of the nodules
            volume% - list with the percentage volumes of the nodules
            larg_diam - list with the largest diameters of the nodules
            Feret_diam - list with the feret diameters of the nodules


    Returns:
        values_dic_out (dictionary):
            'volume' : mean and std of the volume
            'volume extremes' : largest and smallest value of the volume
            'volume%' : mean and the std of the volume percentage
            'larg_diam' : mean and the std of the largest diameters
            'larg_diam_extremes' : largest and smallest value of the diameters
            'Feret diam' :  mean and the std of the Feret diameters
            'Feret diam' : largest and smallest value of the Feret diameters

    """

    # ------- Calculate stats
    stats_dic_out={'volume':[0,0],# mean first, std second
                   'volume extremes':[0,0], #largest and smallest
                   'volume%':[0,0],  # mean first, std second
                   'larg_diam':[0,0],# mean first, std second
                   'larg_diam_extremes':[0,0], #largest and smallest
                   'Feret diam':[0,0], # mean first, std second
                   'Feret diam extremes':[0,0]} #largest and smallest

    stats_dic_out['volume'][0]=df.mean(pd.Series(values_dic['volume']))
    stats_dic_out['volume'][1]=df.std(pd.Series(values_dic['volume']))
    stats_dic_out['volume extremes'][0] = max(values_dic['volume'])
    stats_dic_out['volume extremes'][1] = min(values_dic['volume'])

    stats_dic_out['volume%'][0]=df.mean(pd.Series(values_dic['volume%']))
    stats_dic_out['volume%'][1]=df.std(pd.Series(values_dic['volume%']))

    stats_dic_out['larg_diam'][0]=df.mean(pd.Series(values_dic['larg_diam']))
    stats_dic_out['larg_diam'][1]=df.std(pd.Series(values_dic['larg_diam']))
    stats_dic_out['larg_diam_extremes'][0] = max(values_dic['larg_diam'])
    stats_dic_out['larg_diam_extremes'][1] = min(values_dic['larg_diam'])

    stats_dic_out['Feret diam'][0]=df.mean(pd.Series(values_dic['Feret diam']))
    stats_dic_out['Feret diam'][1]=df.std(pd.Series(values_dic['Feret diam']))
    stats_dic_out['Feret diam extremes'][0] = max(values_dic['Feret diam'])
    stats_dic_out['Feret diam extremes'][1] = min(values_dic['Feret diam'])

    return stats_dic_out
