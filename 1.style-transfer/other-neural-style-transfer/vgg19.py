#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-29 下午3:17
# @Author  : viekie
# @Site    : 
# @File    : vgg19.py.py
# @Software: PyCharm


import tensorflow as tf
import numpy as np
import scipy.io


VGG_NET_LAYERS=(
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
)


def load_vgg_net(path):
    if path is None:
        raise ValueError('vgg model path should not be none')
    data = scipy.io.loadmat(path)
    if not all(i in data for i in ('layers', 'classes', 'normalization')):
        raise ValueError('vgg model file error')

    mean = data['normalization'][0][0][0]
    mean_normal = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    return weights, mean_normal


def preload_vgg_net(weights, input_image, pooling):
    net = {}
    current_input = input_image

    for i, name in enumerate(VGG_NET_LAYERS):
        oper = name[:4]

    if 'conv' == oper:
        weight, bias = weights[i][0][0][0][0]
        kernels = np.transpose(weight, (1, 0, 2, 3))
        bias = bias.reshape(-1)
        current_input = _conv_layer(current_input, weights, bias)
    elif 'relu' == oper:
        current_input = tf.nn.relu(current_input)
    elif 'pool' == oper:
        current_input = _pooling_layer(current_input, pooling)
    net[name] = current_input

    assert(len(net) == len(VGG_NET_LAYERS))
    return net


def _conv_layer(input, weight, bias):
    return tf.nn.conv2d(input, tf.constant(weight), strides=(1, 1, 1, 1), padding='SAME') + bias


def _pooling_layer(input, pooling):
    if 'avg' == pooling:
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


def preprocess(image, mean):
    return image - mean


def unprocess(image, mean):
    return image + mean


if __name__ == '__main__':
    load_vgg_net('./imagenet-vgg-verydeep-19.mat')