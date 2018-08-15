"""Evaluates a model against examples from a .npy file as specified
   in config.json"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np
import tensorflow as tf
from model import Model
from tensorflow.examples.tutorials.mnist import input_data


def run_attack(checkpoint, x_adv, epsilon, results_dir):
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  model = Model()

  saver = tf.train.Saver()

  num_eval_examples = 10000
  eval_batch_size = 64

  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr = 0

  x_nat = mnist.test.images
  l_inf = np.amax(np.abs(x_nat - x_adv))

  if l_inf > epsilon + 0.0001:
    print('maximum perturbation found: {}'.format(l_inf))
    print('maximum perturbation allowed: {}'.format(epsilon))
    return

  y_pred = []
  logits = []
  y_true = []

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, checkpoint)

    # Iterate over the samples batch-by-batch
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = x_adv[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      dict_adv = {model.x_input: x_batch,
                  model.y_input: y_batch}
      cur_corr, y_pred_batch, logits_batch = sess.run(
        [model.num_correct, model.y_pred, model.pre_softmax],
        feed_dict=dict_adv)

      total_corr += cur_corr
      y_pred.append(y_pred_batch)
      logits.append(logits_batch)
      y_true.append(y_batch)

  accuracy = total_corr / num_eval_examples

  print('Accuracy: {:.2f}%'.format(100.0 * accuracy))
  y_pred = np.concatenate(y_pred, axis=0)
  np.save(os.path.join(results_dir, 'pred.npy'), y_pred)

  y_true = np.concatenate(y_true, axis=0)
  np.save(os.path.join(results_dir, 'y_true.npy'), y_true)

  logits = np.concatenate(logits, axis=0)
  np.save(os.path.join(results_dir, 'logits.npy'), logits)

  print('Output saved at %s' % results_dir)


if __name__ == '__main__':
  import json

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_dir = config['model_dir']

  checkpoint = tf.train.latest_checkpoint(model_dir)
  x_adv = np.load(os.path.join(config['results_dir'], config['store_adv_path']))

  if checkpoint is None:
    print('No checkpoint found')
  elif x_adv.shape != (10000, 784):
    print('Invalid shape: expected (10000,784), found {}'.format(x_adv.shape))
  elif np.amax(x_adv) > 1.0001 or \
          np.amin(x_adv) < -0.0001 or \
      np.isnan(np.amax(x_adv)):
    print('Invalid pixel range. Expected [0, 1], found [{}, {}]'.format(
      np.amin(x_adv),
      np.amax(x_adv)))
  else:
    run_attack(checkpoint, x_adv, config['epsilon'], config['results_dir'])
