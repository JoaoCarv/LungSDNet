# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

Script that plots the size distibution of the dataset.

Required changes are:
- path of the csv with the sizes, 'nodules_sizes.csv' (path)
"""

import pandas as pd
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)



path = r'..\create_patches\nodules_sizes.csv'

df_sizes_all = pd.read_csv(path)
# df_sizes_all.rename(columns={'nodule_size (mm)':'test'})
print(df_sizes_all.head())
df_sizes_mm = df_sizes_all['nodule_size (mm)']

# if only considering lesions larger then 6 mm
# df_sizes_mm = df_sizes_mm[df_sizes_mm[:] > 6]
df_final = df_sizes_mm.rename('lesion size (mm)')

sns.set(font_scale = 3,font='serif')


sns.set_context("paper")
sns.set_style("ticks")
# sns.set_style('dark')
plt.close()

a = sns.distplot(df_final,
             hist_kws={"linewidth": 2,
                       "color": "cornflowerblue",
                       "histtype":'bar'
                       },
             kde_kws={"linewidth": 1.8,"bw":0.5,
                      "shade":False,
                      "color":'b'})


plt.show()
plt.xlabel('Lesion Size (mm)',fontsize=30)
plt.ylabel('Fraction per lesion size',fontsize=30)
plt.tick_params(labelsize=30)
rc('xtick',labelsize=15)
