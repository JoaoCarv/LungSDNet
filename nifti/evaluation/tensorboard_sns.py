# -*- coding: utf-8 -*-
"""
@author: João B. Sá Carvalho (jbsa.carvalho@gmail.com)
Project: DeepLearning Application for lung nodule Segmentation

2018/19 @ Champalimaud Foundation/Instituto de Telecomunicações

CODE:
    Script to redraw the plots during the model training. Given the csv files
    extracted from tensorboard it builds a new plot validation and training.

    paths:
        path_dir: directory containing the csv files
        n_train_acc: csv of the acc during training
        n_val_acc: csv of the acc during validation
        n_train_closs: csv of the loss without w_penalization during training
        n_val_closs: csv of the loss without w_penalization during validation
        n_train_loss: csv of the loss with w_penalization during training
        n_val_loss: csv of the loss with w_penalization during vlaidation

"""
from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns
import pandas as pd
import os
from matplotlib import rc


color_code_1 = '#4169E1'
color_code_2 = '#cc3300'
smooth_space = 10
epoch = 1900/300 #iterations
path_dir = r''
# if loss
loss = True


# Files names
n_train_acc = 'run_logs_0_training-tag-worker_0_accuracy.csv'
n_val_acc = 'run_logs_0_validation-tag-worker_0_accuracy.csv'
n_train_closs = 'run_logs_0_training-tag-worker_0_class_loss.csv'
n_val_closs = 'run_logs_0_validation-tag-worker_0_class_loss.csv'
n_train_loss = 'run_logs_0_training-tag-worker_0_loss.csv'
n_val_loss = 'run_logs_0_validation-tag-worker_0_loss.csv'

def smooth_curve(x_raw,y_raw):
    x_smooth = []
    y_smooth = []
    for i in range(0, len(x_raw), smooth_space):
        x_smooth.append(x_raw[i])
        y_smooth.append(sum(y_raw[i:i+smooth_space]) / float(smooth_space))

    x_smooth.append(x_raw[-1])
    y_smooth.append(y_raw[-1])

    return x_smooth, y_smooth
#

# ---------GET DATA
# path
path_train = os.path.join(path_dir,n_train_acc)
path_valid = os.path.join(path_dir,n_val_acc)

df_train = pd.read_csv(path_train)
df_valid = pd.read_csv(path_valid)

x_train = df_train['Step'].values.tolist()
y_train = df_train['Value'].values.tolist()
x_train = [x / epoch for x in x_train]

x_valid = df_valid['Step'].values.tolist()
y_valid = df_valid['Value'].values.tolist()
x_valid = [x / epoch for x in x_valid]

if loss is True:
    path_train_closs = os.path.join(path_dir, n_train_closs)
    path_valid_closs = os.path.join(path_dir, n_val_closs)
    path_train_loss = os.path.join(path_dir, n_train_loss)
    path_valid_loss = os.path.join(path_dir, n_val_loss)

    df_train_closs = pd.read_csv(path_train_closs)
    df_valid_closs = pd.read_csv(path_valid_closs)
    df_train_loss = pd.read_csv(path_train_loss)
    df_valid_loss = pd.read_csv(path_valid_loss)

    x_train_closs = df_train_closs['Step'].values.tolist()
    y_train_closs = df_train_closs['Value'].values.tolist()
    x_train_closs = [x / epoch for x in x_train_closs]

    x_valid_closs = df_valid_closs['Step'].values.tolist()
    y_valid_closs = df_valid_closs['Value'].values.tolist()
    x_valid_closs = [x / epoch for x in x_valid_closs]

    x_train_loss = df_train_loss['Step'].values.tolist()
    y_train_loss = df_train_loss['Value'].values.tolist()
    x_train_loss = [x / epoch for x in x_train_loss]

    x_valid_loss = df_valid_loss['Step'].values.tolist()
    y_valid_loss = df_valid_loss['Value'].values.tolist()
    x_valid_loss = [x / epoch for x in x_valid_loss]


# ---------Prepare curves
# smooth curve train

x_train_smooth, y_train_smooth = smooth_curve(x_train,y_train)
x_valid_smooth, y_valid_smooth = smooth_curve(x_valid,y_valid)

if loss is True:
    x_train_closs_smooth,y_train_closs_smooth = smooth_curve(x_train_closs,
                                                             y_train_closs)
    x_valid_closs_smooth,y_valid_closs_smooth = smooth_curve(x_valid_closs,
                                                             y_valid_closs)
    x_train_loss_smooth,y_train_loss_smooth = smooth_curve(x_train_loss,
                                                             y_train_loss)
    x_valid_loss_smooth,y_valid_loss_smooth = smooth_curve(x_valid_loss,
                                                             y_valid_loss)



# --------Plots
# style
#sns.set(style="darkgrid")
#sns.set_context("paper")
sns.set_context("paper")
sns.set_style("ticks")
palette_b = sns.color_palette("Blues")
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
rc('text', usetex=True)


plt.close('all')

# 1. ACCURACY
plt.figure(1)
label='histogram of undetected nodules'
plt.plot(x_train,
         y_train,
         color=colors.to_rgba(color_code_1, alpha=0.4))
plt.plot(x_train_smooth,
         y_train_smooth,
         color=color_code_1,
         linewidth=1.5,
         label='trainings')
plt.plot(x_valid,
         y_valid,
         color=colors.to_rgba(color_code_2, alpha=0.4))
plt.plot(x_valid_smooth,
         y_valid_smooth,
         color=color_code_2,
         linewidth=1.5,
         label='validation')

# extras
plt.tick_params(labelsize=25)
plt.xlabel('Epoch',fontsize=35)
plt.legend(fontsize=20)
plt.ylabel('Accuracy', fontsize = 35)

if loss is True:
    # 2. Class loss
    plt.figure(2)
    label='histogram of undetected nodules'
    plt.plot(x_train_closs,
             y_train_closs,
             color=colors.to_rgba(color_code_1, alpha=0.4))
    plt.plot(x_train_closs_smooth,
             y_train_closs_smooth,
             color=color_code_1,
             linewidth=1.5,
             label='trainings')
    plt.plot(x_valid_closs,
             y_valid_closs,
             color=colors.to_rgba(color_code_2, alpha=0.4))
    plt.plot(x_valid_closs_smooth,
             y_valid_closs_smooth,
             color=color_code_2,
             linewidth=1.5,
             label='validation')

    # extras
    plt.tick_params(labelsize=25)
    plt.xlabel('Epoch',fontsize=35)
    plt.legend(fontsize=20)
    plt.ylabel('Cross-entropy loss', fontsize = 23)

    # 3. Full loss
    plt.figure(3)
    label='histogram of undetected nodules'
    plt.plot(x_train_loss,
             y_train_loss,
             color=colors.to_rgba(color_code_1, alpha=0.4))
    plt.plot(x_train_loss_smooth,
             y_train_loss_smooth,
             color=color_code_1,
             linewidth=1.5,
             label='trainings')
    plt.plot(x_valid_loss,
             y_valid_loss,
             color=colors.to_rgba(color_code_2, alpha=0.4))
    plt.plot(x_valid_loss_smooth,
             y_valid_loss_smooth,
             color=color_code_2,
             linewidth=1.5,
             label='validation')

    # extras
    plt.tick_params(labelsize=25)
    plt.xlabel('Epoch',fontsize=35)
    plt.legend(fontsize=20)
    plt.ylabel('Cross-entropy + $\ell_1$ + $\ell_2$ loss', fontsize = 23)



plt.show()
