import itertools
import os

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from unrestricted_advex.mnist_baselines import mnist_convnet, mnist_utils
from unrestricted_advex import attacks, eval_kit


def test_two_class_mnist():
  model_dir = '/tmp/two-class-mnist/test'
  batch_size = 128
  dataset_total_n_batches = 1  # use a subset of the data
  train_batches = 20

  # Train a little MNIST classifier from scratch
  mnist = mnist_utils.mnist_dataset(one_hot=False)
  num_datapoints = batch_size * dataset_total_n_batches
  next_batch_iter = mnist_utils.two_class_iter(
    mnist.train.images, mnist.train.labels,
    num_datapoints=num_datapoints, batch_size=batch_size,
    label_scheme='one_hot', cycle=True)

  mnist_utils.train_mnist(model_dir, lambda: next(next_batch_iter),
                          train_batches, "vanilla", save_every=(train_batches-1),
                          print_every=10)

  # Test it on small attacks *on the training set*
  model_fn = mnist_utils.np_two_class_mnist_model(model_dir)

  two_class_iter = mnist_utils.two_class_iter(
    mnist.train.images, mnist.train.labels,
    num_datapoints=num_datapoints, batch_size=batch_size)

  attack = attacks.NullAttack()
  _, _, correct, _ = eval_kit.run_attack(
    model_fn, two_class_iter, attack)

  # not sure exactly how much fudge factor is needed
  assert np.sum(correct) >= (num_datapoints - 1)
