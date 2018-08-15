"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Model(object):
  def __init__(self):
    # first convolutional layer
    self.W_conv1 = _weight_variable([5, 5, 1, 32])
    self.b_conv1 = _bias_variable([32])

    # second convolutional layer
    self.W_conv2 = _weight_variable([5, 5, 32, 64])
    self.b_conv2 = _bias_variable([64])

    # first fully connected layer
    self.W_fc1 = _weight_variable([7 * 7 * 64, 1024])
    self.b_fc1 = _bias_variable([1024])

    # output layer
    self.W_fc2 = _weight_variable([1024, 10])
    self.b_fc2 = _bias_variable([10])

  def __call__(self, x_input):
    x_image = tf.reshape(x_input, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(_conv2d(x_image, self.W_conv1) + self.b_conv1)
    h_pool1 = _max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(_conv2d(h_pool1, self.W_conv2) + self.b_conv2)
    h_pool2 = _max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    self.h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

    return tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2


def _weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def _bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def _conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def _max_pool_2x2(x):
  return tf.nn.max_pool(x,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
