#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-30 下午3:30
# @Author  : viekie
# @Site    : 
# @File    : main.py.py
# @Software: PyCharm


import os
import numpy as np
import scipy.misc
import math
import argparse as parser
from PIL import Image


def arg_parse():
    psr = parser.ArgumentParser(description='define args')
    psr.add_argument('--content_image', type=str, default='./input/content/input.jpg', help='set content image path')
    psr.add_argument('--style_image', type=str, default='./input/style/style.jpg', help='set style image path')
    psr.add_argument('--output_dir', type=str, help='set output image path')
    psr.add_argument('--iteration', type=int, default=1000, help='iterator number')
    psr.add_argument('--print_iteration', type=int, default=10, help='print iterator')
    psr.add_argument('--check_point', type=str, default='./check_point', help='check point path')
    psr.add_argument('--check_point_iterator', type=int, default=100, help='check point path')
    psr.add_argument('--width', type=int, help='image width')
    psr.add_argument('--style_scales', type=float, default=1.0, help='one or more style scales')
    psr.add_argument('--network', type=str, default='./imagenet-vgg-verydeep-19.mat')
    psr.add_argument('--content-weight-blend', type=float,
                     help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)',
                     default=1.0)
    psr.add_argument('--content-weight', type=float, dest='content_weight',
                     help='content weight (default %(default)s)', default=5.0)
    psr.add_argument('--style-weight', type=float, dest='style_weight',
                     help='style weight (default %(default)s)', metavar='STYLE_WEIGHT', default=5.0)
    psr.add_argument('--style-layer-weight-exp', type=float, dest='style_layer_weight_exp',
                     help='''style layer weight exponentional increase - weight(layer<n+1>) 
                     = weight_exp*weight(layer<n>) (default %(default)s)''',
                     metavar='STYLE_LAYER_WEIGHT_EXP', default=1)
    psr.add_argument('--style-blend-weights', type=float,
                     dest='style_blend_weights', help='style blending weights')
    psr.add_argument('--tv-weight', type=float,
                     dest='tv_weight', help='total variation regularization weight (default %(default)s)',
                     metavar='TV_WEIGHT', default=1e2)
    psr.add_argument('--learning-rate', type=float,
                     dest='learning_rate', help='learning rate (default %(default)s)',
                     metavar='LEARNING_RATE', default=1e1)
    psr.add_argument('--beta1', type=float,
                     dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
                     metavar='BETA1', default=0.9)
    psr.add_argument('--beta2', type=float,
                     dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
                     metavar='BETA2', default=0.999)
    psr.add_argument('--eps', type=float,
                     dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
                     metavar='EPSILON', default=1e08)
    psr.add_argument('--initial',
                     dest='initial', help='initial image',
                     metavar='INITIAL')
    psr.add_argument('--initial-noiseblend', type=float,
                     dest='initial_noiseblend',
                     help='ratio of blending initial image with normalized noise (if no initial image specified, content image is used) (default %(default)s)',
                     metavar='INITIAL_NOISEBLEND')
    psr.add_argument('--preserve-colors', action='store_true',
                     dest='preserve_colors',
                     help='style-only transfer (preserving colors) - if color transfer is not needed')
    psr.add_argument('--pooling',
                     dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)',
                     metavar='POOLING', default='max')
    return psr


if __name__ == '__main__':
    parser = arg_parse()
    args = parser.parse_args()

    if args.network is None or os.path.exists(args.network):
        raise ValueError('vgg19 model file not exist')

    if args.content_image is None or os.path.exists(args.content_image):
        raise ValueError('content image not exist')
    elif:
        content_image = imread
    if args.style_image is None or os.path.exists(args.style_image):
        raise ValueError('style image not exist')
    elif:
        style_images = [imread(style) for style in args.style_image]
