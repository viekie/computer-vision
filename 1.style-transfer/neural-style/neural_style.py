#!/usr/bin/env python
# -*- coding: utf8 -*-
# Power by viekie2017-09-27 09:27:15


import argparse as parser
import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.io
import os


mean_value = np.array([123, 117, 104]).reshape(1, 1, 1, 3)
content_layer = [('conv4_2', 1.0)]
style_layer = [('conv1_1', 1.), ('conv2_1', 1.), ('conv3_1', 1.),
               ('conv4_1', 1.), ('conv5_1', 1.)]


##
# @brief define arg parser
#
# @return
def args_parser():
    psr = parser.ArgumentParser()
    psr.add_argument('--content_image_width', type=int, default=800,
                     help='content image\'s width')
    psr.add_argument('--content_image_height', type=int, default=600,
                     help='content image\'s height')
    psr.add_argument('--content_image', type=str,
                     default='./input/content/input.jpg',
                     help='input content image full file name')
    psr.add_argument('--style_image', type=str,
                     default='./input/style/style.jpg',
                     help='input style image full file name')
    psr.add_argument('--output_dir', type=str,
                     default='./result/', help='output image dir')
    psr.add_argument('--vgg_model', type=str,
                     default='imagenet-vgg-verydeep-19.mat',
                     help='vgg model')
    psr.add_argument('--ini_noise_ratio', type=float, default=0.7,
                     help='noise ratio')
    psr.add_argument('--style_strength', type=int, default=500,
                     help='style strength')
    psr.add_argument('--iteration', type=int, default=5000,
                     help='times of iteration')
    args = psr.parse_args()
    return args


args = args_parser()


##
# @brief  read image
#
# @param path
#
# @return
def read_img(path):
    if path is None:
        raise ValueError()
    image = scipy.misc.imread(path)
    image = image[np.newaxis, :, :, :]

    image = image - mean_value
    return image


##
# @brief write image
#
# @param path
# @param image
#
# @return
def write_img(path, image):
    image = image + mean_value
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


def build_content_loss(p, x):
    M = p.shape[1] * p.shape[2]
    N = p.shape[3]
    loss = (1./(2*N**0.5*M**0.5)) * tf.reduce_sum(tf.pow((x-p), 2))
    return loss


def build_style_loss(a, x):
    M = a.shape[1] * a.shape[2]
    N = a.shape[3]
    A = gram_matrix_val(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1.0/(4*N**2*M**2)) * tf.reduce_sum(tf.pow((G-A), 2))
    return loss


def gram_matrix(x, area, depth):
    x1 = tf.reshape(x, (area, depth))
    g = tf.matmul(tf.transpose(x1), x1)
    return g


def gram_matrix_val(x, area, depth):
    x1 = x.reshape(area, depth)
    g = np.dot(x1.T, x1)
    return g


def main():
    net = build_vgg19(args.vgg_model)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        noise_img = np.random.uniform(-20, 20,
                                      (1, args.content_image_height,
                                       args.content_image_width,
                                       3)).astype('float32')
        content_img = read_img(args.content_image)
        style_img = read_img(args.style_image)

        sess.run([net['input'].assign(content_img)])
        cost_content = sum(map(lambda l:
                               l[1] * build_content_loss(sess.run(net[l[0]]),
                                                         net[l[0]]),
                               content_layer))

        sess.run([net['input'].assign(style_img)])
        cost_style = sum(map(lambda l:
                             l[1] * build_style_loss(sess.run(net[l[0]]),
                                                     net[l[0]]),
                             style_layer))

        cost_total = cost_content + args.style_strength * cost_style
        trainer = tf.train.AdamOptimizer(2.0)
        optimizer = trainer.minimize(cost_total)
        sess.run(tf.initialize_all_variables())
        sess.run(net['input'].assign(args.ini_noise_ratio * noise_img +
                                     (1.0-args.ini_noise_ratio)*content_img))
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        # sess.run(tf.global_variables_initializer())
        for i in range(args.iteration):
            sess.run(optimizer)
            result_img = sess.run(net['input'])
            print('iterator: ', i, ', cost:', cost_total.eval(session=sess))
            if i % 10 == 0:
                write_img(os.path.join(args.output_dir,
                                       '%s.png' % (str(i).zfill(4))),
                          result_img)
        write_img(os.path.join(args.output_dir, 'final.png'), result_img)


##
# @brief get weights and bias from vgg_layers
#
# @param vgg_layers
# @param i
#
# @return
def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][0][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][0][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias


##
# @brief conv or pool operator
#
# @param oper
# @param input
# @param weights_bias
#
# @return
def build_net(oper, input, weights_bias=None):
    if oper == 'conv':
        conv = tf.nn.conv2d(input, weights_bias[0],
                            strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(conv + weights_bias[1])
    elif oper == 'pool':
        return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


##
# @brief from input vgg model to build vgg net
#
# @param path vgg model file path
#
# @return
def build_vgg19(path):
    net = {}
    if path is None:
        raise ValueError()
    vgg_raw = scipy.io.loadmat(path)
    vgg_layers = vgg_raw['layers'][0]
    net['input'] = tf.Variable(np.zeros((1, args.content_image_height,
                                         args.content_image_width,
                                         3)).astype('float32'))
    # layer 1
    net['conv1_1'] = build_net('conv', net['input'],
                               get_weight_bias(vgg_layers, 0))
    net['conv1_2'] = build_net('conv', net['conv1_1'],
                               get_weight_bias(vgg_layers, 2))
    net['pool1'] = build_net('pool', net['conv1_2'])

    # layer2
    net['conv2_1'] = build_net('conv', net['pool1'],
                               get_weight_bias(vgg_layers, 5))
    net['conv2_2'] = build_net('conv', net['conv2_1'],
                               get_weight_bias(vgg_layers, 7))
    net['pool2'] = build_net('pool', net['conv2_2'])

    # layer3
    net['conv3_1'] = build_net('conv', net['pool2'],
                               get_weight_bias(vgg_layers, 10))
    net['conv3_2'] = build_net('conv', net['conv3_1'],
                               get_weight_bias(vgg_layers, 12))
    net['conv3_3'] = build_net('conv', net['conv3_2'],
                               get_weight_bias(vgg_layers, 14))
    net['conv3_4'] = build_net('conv', net['conv3_3'],
                               get_weight_bias(vgg_layers, 16))
    net['pool3'] = build_net('pool', net['conv3_4'])

    # layer4
    net['conv4_1'] = build_net('conv', net['pool3'],
                               get_weight_bias(vgg_layers, 19))
    net['conv4_2'] = build_net('conv', net['conv4_1'],
                               get_weight_bias(vgg_layers, 21))
    net['conv4_3'] = build_net('conv', net['conv4_2'],
                               get_weight_bias(vgg_layers, 23))
    net['conv4_4'] = build_net('conv', net['conv4_3'],
                               get_weight_bias(vgg_layers, 25))
    net['pool4'] = build_net('pool', net['conv4_4'])

    # layer5
    net['conv5_1'] = build_net('conv', net['pool4'],
                               get_weight_bias(vgg_layers, 28))
    net['conv5_2'] = build_net('conv', net['conv5_1'],
                               get_weight_bias(vgg_layers, 30))
    net['conv5_3'] = build_net('conv', net['conv5_2'],
                               get_weight_bias(vgg_layers, 32))
    net['conv5_4'] = build_net('conv', net['conv5_3'],
                               get_weight_bias(vgg_layers, 34))
    net['pool5'] = build_net('pool', net['conv5_4'])

    return net


if __name__ == '__main__':
    main()
