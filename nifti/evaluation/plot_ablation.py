# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    Script to build a plot with either box plots or violin plots that have
    condensed the results of all architectures tested in the ablation study.

    paths:
        path_dir: directory that has the results of all the inferences of the
        ablation studies already evaluated with "evaluation_classes_ablation"
"""

import os
from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns
import pandas as pd
from matplotlib import rc


path_dir = r''
path_no_conv = os.path.join(path_dir,r'OUT_NO_CONV')
path_yes_conv = os.path.join(path_dir,r'OUT_YES_CONV')
path_yes_res = os.path.join(path_dir,r'OUT_YES_RES')

# Style definions
sns.set_context("paper")
sns.set_style("ticks")
palette_b = sns.color_palette("Blues")
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
rc('text', usetex=True)


# Load
df_no_conv = pd.read_csv(path_no_conv + '\_eval.csv')
df_yes_conv = pd.read_csv(path_yes_conv + '\_eval.csv')
df_yes_res = pd.read_csv(path_yes_res + '\_eval.csv')


# Change labels
df_no_conv_2 = df_no_conv[df_no_conv['label']=='training']
df_no_conv_3 = df_no_conv[df_no_conv['label']=='test']
df_no_conv_2['label'] = 'test'
df_no_conv_3['label'] = 'training'
df_no_conv = pd.concat([df_no_conv_3,df_no_conv_2])

plt.close('all')

# ------------- violin plot
#plt.figure(1)
#ax = sns.violinplot(x='iteration', y='acc',
#                    hue="label",data=df_no_conv,
#                    split=True)
#                    #palette="muted")
#plt.figure(2)
#ax = sns.violinplot(x='iteration', y='acc',
#                    hue="label",data=df_yes_conv,
#                    split=True)
#                    #palette="muted")
#plt.figure(3)
#ax = sns.violinplot(x='iteration', y='acc',
#                    hue="label",data=df_yes_res,
#                    split=True)
#                    #palette="muted")


# ------------- box plot
colors_b = ["light blue","cool blue"]
palette_blues = sns.xkcd_palette(colors_b)
plt.figure(1)
ax = sns.boxplot(x='iteration', y='acc',
                    hue="label",data=df_no_conv,
                    palette=palette_blues)

ax.tick_params(labelsize=20)
ax.set_xlabel('Iteration',fontsize = 30)
ax.set_ylabel('Accuracy',fontsize = 30)
ax.legend(fontsize=14,fancybox=True)

plt.figure(2)
ax = sns.boxplot(x='iteration', y='acc',
                    hue="label",data=df_yes_conv,
                    palette=palette_blues)

ax.tick_params(labelsize=20)
ax.set_xlabel('Iteration',fontsize = 30)
ax.set_ylabel('Accuracy',fontsize = 30)
ax.legend(fontsize=14,fancybox=True)
#
plt.figure(3)
ax = sns.boxplot(x='iteration', y='acc',
                    hue="label",data=df_yes_res,
                    palette=palette_blues)
                    #palette="muted")
ax.tick_params(labelsize=20)
ax.legend(fontsize=14,fancybox=True)
ax.set_xlabel('Iteration',fontsize = 30)
ax.set_ylabel('Accuracy',fontsize = 30)
