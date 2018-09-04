import os
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from unrestricted_advex.mnist_baselines import mnist_convnet
from unrestricted_advex.mnist_baselines.train_two_class_mnist \
  import train_two_class_mnist
from unrestricted_advex.mnist_baselines.evaluate_two_class_mnist \
  import two_class_mnist_iter, np_two_class_mnist_model
from unrestricted_advex import attacks, eval_kit


def test_two_class_mnist():
  model_dir = '/tmp/two-class-mnist/test'
  batch_size = 128
  train_batches = 1
  test_batches = 1

  # Train a little MNIST classifier from scratch
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  next_batch_fn = mnist.train.next_batch
  train_two_class_mnist(model_dir, next_batch_fn, batch_size, train_batches,
                        "vanilla")

  # Test it on small attacks
  model_fn = np_two_class_mnist_model(model_dir),
  two_class_iter = two_class_mnist_iter(num_datapoints=128, batch_size=128)

  attack = attacks.NullAttack()
  logits, labels, correct, x_adv = eval_kit.run_attack(
    model_fn, two_class_iter, attack)
