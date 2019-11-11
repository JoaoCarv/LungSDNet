# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    Script to evaluate the lesions that were failed to be detected in the test
    set. A plot that incorporates the size distribution in the training set
    and the size distribution in the failed attempts is incorporated.

    paths:
        path_dir: path of the results output (the output file is assumed to
        be named '_niftyet_out.csv')
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


path_dir = r'C:\JFCImportantes\Universidade\Thesis\RESULTS\try3'
path_out_csv = os.path.join(path_dir,r'out\_niftynet_out.csv')
path_truth_csv = os.path.join(path_dir,'nodules_class.csv')
path_split_csv = os.path.join(path_dir,'dataset_split_class.csv')


# Load file
df_out = pd.read_csv(path_out_csv, names=['id', 'class'])
df_truth = pd.read_csv(path_truth_csv,
                       names=['id','class','size','lx1','lx2','lx3'])
df_split = pd.read_csv(path_split_csv,
                       names=['id','label'])

# Curation
df_split_train= df_split[df_split['label']=='training']


df_truth = df_truth[df_truth['class']==1] #remove random from truth
df_truth_test = df_truth[df_truth['id'].isin(df_out['id'])] #select only from testset
df_truth_training = df_truth[df_truth['id'].isin(df_split_train['id'])] #select only from train

df_out_non_rand = df_out[df_out['id'].isin(df_truth_test['id'])] #remove rand from out
df_out_fail= df_out_non_rand[df_out_non_rand['class']==0] # find failed
df_truth_failed = df_truth_test[df_truth_test['id'].isin(df_out_fail['id'])]


# Get Ranges
hist_out=plt.hist(df_truth_failed['size'],bins=5)
plt.close()


ranges = hist_out[1]
nbs_failed = hist_out[0]


def n_per_size(df_truth,ranges):
    nbs=[]

    # First range
    nb_range_1 = df_truth[df_truth['size']>ranges[1]]
    nbs.append(nb_range_1.shape[0])

    # Second range
    nb_range_2 = df_truth[df_truth['size']>ranges[1]]
    nb_range_2 = nb_range_2[nb_range_2['size']<ranges[2]]
    nbs.append(nb_range_2.shape[0])

    # Third range
    nb_range_3 = df_truth[df_truth['size']>ranges[2]]
    nb_range_3 = nb_range_3[nb_range_3['size']<ranges[3]]
    nbs.append(nb_range_3.shape[0])


    # Fourth range
    nb_range_4 = df_truth[df_truth['size']>ranges[3]]
    nb_range_4 = nb_range_4[nb_range_4['size']<ranges[4]]
    nbs.append(nb_range_4.shape[0])

    # Fifth range
    nb_range_5 = df_truth[df_truth['size']>ranges[4]]
    nbs.append(nb_range_5.shape[0])

    return nbs

nbs_training = n_per_size(df_truth_training,ranges)
nbs_test = n_per_size(df_truth_test,ranges)
#perc_test = np.array(nbs_test)/sum(nbs_test)


# training size distribution
perc_training = np.array(nbs_training)/sum(nbs_training)


# -------- Plots
sns.set(font_scale = 3,font='serif')


sns.set_context("paper")
sns.set_style("ticks")

# Histogram failed
weights_into=np.zeros_like(df_truth_failed['size']) + 1. / df_truth_failed['size'].size
ax = sns.distplot(df_truth_failed['size'],
                  kde=False,
                  bins=5,
                  label='test',
                  hist_kws={'weights':weights_into}
                  )#,hist_kws={"density":None,"normed":True})
plt.tick_params(labelsize=20)
plt.xlabel('Lesion Size (mm)',fontsize=30)
#ax.legend()
plt.ylabel('Failed fraction per size', fontsize = 30)
#ax.yaxis.set_major_formatter(PercentFormatter(1))
ax2 = plt.twinx()


# Scatter percentages
perc_failed = np.array(nbs_failed)/nbs_test

sizes_mid = []
mid = (ranges[2]-ranges[1])/2
for i in range(len(ranges)-1):
    sizes_mid.append(ranges[i]+mid)
#plt.plot(sizes_mid,perc_failed,color='cornflowerblue',label='histogram of undetected nodules')
ax2.plot(sizes_mid,perc_training,'x',dashes=[2,2],color='red',label='size distribution in training')
plt.tick_params(labelsize=20)
plt.show()

ax2.yaxis.set_major_formatter(PercentFormatter(1))
ax2.set_ylabel('Percentage per size fraccion',fontsize = 30)
#sns.lineplot(data=df.column2, color="b", ax=ax2)
ax2.legend()
