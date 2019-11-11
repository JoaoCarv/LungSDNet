# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    beep_sound: plays a sound to alert
    t_current: gives the current time
"""

import time
import csv


def beep_sound(repetitions=1):
    """beep sound to use at the relevant sections of the code"""
    from winsound import Beep
    for i in range(repetitions):
        Beep(500, 500)
        Beep(1000, 500)
        Beep(750, 500)


def t_curent():
    return time.time()


def append2csv(csv_path, row):
    """
    Appends the information in row to the csv file
    defined in csv_path.
    csv_path : string
    row : list

    """

    with open(csv_path, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
