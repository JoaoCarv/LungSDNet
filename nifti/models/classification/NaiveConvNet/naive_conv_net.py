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


class Naive_Conv_Net(BaseNet):
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
                 name='naive_conv_net'):

        BaseNet.__init__(self,
                         num_classes=num_classes,
                         name=name)
        self.n_fea = [64, 128, 256, 512, 512, 2, 4]


        net_params = {'padding': 'SAME',
                      'with_bias': True,
                      'with_bn': False,
                      'acti_func': acti_func,
                      'w_initializer': w_initializer,
                      'b_initializer': b_initializer,
                      'w_regularizer': w_regularizer,
                      'b_regularizer': b_regularizer}

        # self.conv_params_1 = {'kernel_size': 5, 'stride': 1}
        self.conv_params = {'kernel_size': 3, 'stride': 1}


        self.deconv_params = {'kernel_size': 2, 'stride': 2}
        self.pooling_params = {'kernel_size': 2, 'stride': 2}

        self.conv_params.update(net_params)
        self.deconv_params.update(net_params)

    def layer_op(self, image, is_training=True, **unused_kwargs):

        # ---Conv Layers
        output_1 = ThreeLayerConv(self.n_fea[0], self.conv_params)(image)
        down_1 = Pooling(func='MAX', **self.pooling_params)(output_1)

        output_2 = ThreeLayerConv(self.n_fea[1], self.conv_params)(down_1)
        down_2 = Pooling(func='MAX', **self.pooling_params)(output_2)

        output_3 = ThreeLayerConv(self.n_fea[2], self.conv_params)(down_2)
        down_3 = Pooling(func='MAX', **self.pooling_params)(output_3)



        # ---FC layers
        FC_1 = FCLayer(n_output_chns=self.n_fea[3],
                       acti_func='leakyrelu',
                       with_bias=True,
                       with_bn=False,
                       w_initializer=self.conv_params.get('w_initializer'),
                       w_regularizer=self.conv_params.get('w_regularizer'))(down_3)

        FC_1_drop = ActiLayer(func='dropout', name='dropout')(FC_1, keep_prob=0.5)

        FC_2 = FCLayer(n_output_chns=self.n_fea[4],
                       acti_func='leakyrelu',
                       with_bias=True,
                       with_bn=False,
                       w_initializer=self.conv_params.get('w_initializer'),
                       w_regularizer=self.conv_params.get('w_regularizer'))(FC_1_drop)
        FC_2_drop = ActiLayer(func='dropout', name='dropout')(FC_2, keep_prob=0.5)


        FC_class = FCLayer(n_output_chns=self.n_fea[5],
                           acti_func='leakyrelu',
                           with_bias=True,
                           with_bn=False,
                           w_initializer=self.conv_params.get('w_initializer'),
                           w_regularizer=self.conv_params.get('w_regularizer'))(FC_2_drop)


        return FC_class


class ThreeLayerConv(TrainableLayer):
    """
    Three convolutional layers, with dropout and residual connections number of output channels are ``n_chns`` for both
    of them.

    --conv--conv--
    """

    def __init__(self, n_chns, conv_params):
        TrainableLayer.__init__(self, name='ThreeConv')
        self.n_chns = n_chns
        self.conv_params = conv_params

    def layer_op(self, input_tensor):
        output_tensor_1 = Conv(self.n_chns, **self.conv_params)(input_tensor)
        output_tensor_drop_1 = ActiLayer(func='dropout', name='dropout')(output_tensor_1, keep_prob=0.9)
        output_tensor_2 = Conv(self.n_chns, **self.conv_params)(output_tensor_drop_1)
        output_tensor_drop_2 = ActiLayer(func='dropout', name='dropout')(output_tensor_2, keep_prob=0.9 )
        output_tensor_3 = Conv(self.n_chns, **self.conv_params)(output_tensor_drop_2)
        output_tensor_drop_3 = ActiLayer(func='dropout', name='dropout')(output_tensor_3, keep_prob=0.9)
        return output_tensor_drop_3
