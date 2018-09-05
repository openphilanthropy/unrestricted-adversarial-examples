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
  train_batches = 200
  test_batches = 10

  # Train a little MNIST classifier from scratch
  mnist = mnist_utils.mnist_dataset()
  images = mnist.train.images[0:dataset_total_n_batches*batch_size]
  labels = mnist.train.labels[0:dataset_total_n_batches*batch_size]

  def get_next_batch_fn():
    idx_cycle = itertools.cycle(range(len(images)))
    def next_batch_fn(batch_size):
      images_batch = np.ndarray((0, 28*28))
      labels_batch = np.ndarray((0, 10))
      for _ in range(batch_size):
        idx = next(idx_cycle)
        images_batch = np.append(images_batch, images[idx,None], axis=0)
        labels_batch = np.append(labels_batch, labels[idx,None], axis=0)
      return (images_batch, labels_batch)
    return next_batch_fn

  next_batch_fn = get_next_batch_fn()
  mnist_utils.train_mnist(model_dir, next_batch_fn, batch_size, train_batches,
                         "vanilla")

  # Test it on small attacks
  model_fn = mnist_utils.np_two_class_mnist_model(model_dir)
  num_datapoints = batch_size * test_batches

  two_class_iter = mnist_utils.two_class_iter(
    mnist.train.images, mnist_utils.one_hot_to_labels(mnist.train.labels),
    num_datapoints=num_datapoints, batch_size=batch_size)

  attack = attacks.NullAttack()
  _, _, correct, _ = eval_kit.run_attack(
    model_fn, two_class_iter, attack)
  assert(np.sum(correct) > (num_datapoints * 0.5))
