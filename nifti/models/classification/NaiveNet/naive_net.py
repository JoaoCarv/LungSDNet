# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.fully_connected import FullyConnectedLayer as FCLayer
from niftynet.network.base_net import BaseNet
from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.base_layer import TrainableLayer, Layer
from niftynet.layer.downsample import DownSampleLayer as Pooling
from niftynet.layer.activation import ActiLayer
from niftynet.layer.elementwise import ElementwiseLayer as ElementWise


class Naive_Net(BaseNet):
    """
    A reimplementation Central Focused Network:
        Wang et al., Central focused convolutional neural networks: Developing
        a data-driven model for lung nodule segmentation PMC 2017
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='leakyrelu',
                 name='naive_net'):

        BaseNet.__init__(self,
                         num_classes=num_classes,
                         name=name)
        self.n_fea = [512, 512, 2]


        net_params = {'padding': 'SAME',
                      'with_bias': True,
                      'with_bn': False,
                      'acti_func': acti_func,
                      'w_initializer': w_initializer,
                      'b_initializer': b_initializer,
                      'w_regularizer': w_regularizer,
                      'b_regularizer': b_regularizer}

        # self.conv_params_1 = {'kernel_size': 5, 'stride': 1}

    def layer_op(self, image, is_training=True, **unused_kwargs):

        # ---FC layers
        FC_1 = FCLayer(n_output_chns=self.n_fea[0],
                       acti_func='leakyrelu',
                       with_bias=True,
                       with_bn=False,
                       w_initializer=self.conv_params.get('w_initializer'),
                       w_regularizer=self.conv_params.get('w_regularizer'))(image)

        FC_1_drop = ActiLayer(func='dropout', name='dropout')(FC_1, keep_prob=0.5)

        FC_2 = FCLayer(n_output_chns=self.n_fea[1],
                       acti_func='leakyrelu',
                       with_bias=True,
                       with_bn=False,
                       w_initializer=self.conv_params.get('w_initializer'),
                       w_regularizer=self.conv_params.get('w_regularizer'))(FC_1_drop)
        FC_2_drop = ActiLayer(func='dropout', name='dropout')(FC_2, keep_prob=0.5)


        FC_class = FCLayer(n_output_chns=self.n_fea[2],
                           acti_func='leakyrelu',
                           with_bias=True,
                           with_bn=False,
                           w_initializer=self.conv_params.get('w_initializer'),
                           w_regularizer=self.conv_params.get('w_regularizer'))(FC_2_drop)


        return FC_class