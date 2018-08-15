"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

VALID_LABELS = [4, 5]
VALID_LABEL_MASK = np.zeros(10, dtype=np.float32)
for label in VALID_LABELS:
  VALID_LABEL_MASK[label] = 1.


class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start

    if loss_func == 'xent':
      y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=model.y_input, logits=model.pre_softmax)
      loss = tf.reduce_sum(y_xent)

    elif loss_func == 'cw':
      true_label_mask = tf.one_hot(model.y_input, 10, dtype=tf.float32)
      correct_logit = tf.reduce_sum(true_label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1 - true_label_mask) * model.pre_softmax, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)

    elif loss_func == 'cw-tcu':
      true_label_mask = tf.one_hot(model.y_input, 10, dtype=tf.float32)

      valid_label_mask = tf.constant(VALID_LABEL_MASK)
      valid_label_mask = tf.Print(valid_label_mask, [valid_label_mask], first_n=1, summarize=1000)

      correct_logit = tf.reduce_sum(true_label_mask * model.pre_softmax, axis=1)

      wrong_label_mask = tf.clip_by_value(valid_label_mask - true_label_mask, 0, 1)
      wrong_label_mask = tf.Print(wrong_label_mask, [wrong_label_mask], first_n=1, summarize=1000)

      wrong_logit = tf.reduce_max(wrong_label_mask * model.pre_softmax, axis=1)

      loss = wrong_logit - correct_logit
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      x += self.a * np.sign(grad)

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 1)  # ensure valid pixel range

    return x


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model()
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = []  # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = os.path.join(config['results_dir'], config['store_adv_path'])
    os.makedirs(config['results_dir'], exist_ok=True)
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))
