"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import shutil
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from adv_mnist.model import Model
from adv_mnist.pgd_attack import LinfPGDAttack
from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir')
parser.add_argument('--train-alp-pgd-steps', default=0, type=int)
args = parser.parse_args()

NOISE_STD = 0.6

with open('config.json') as config_file:
  config = json.load(config_file)
  print("\n\n==== CONFIG ====")
  print(config)
  print("================\n\n")

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()

x_input = tf.placeholder(tf.float32, shape=[None, 784])
y_input = tf.placeholder(tf.int64, shape=[None])
noise_input = tf.placeholder(tf.float32, shape=[None, 784])

model = Model()
logits = model(x_input)
noise_logits = model(x_input + noise_input)

logits_mse = tf.losses.mean_squared_error(logits, noise_logits)

clean_xent = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
  labels=y_input, logits=logits))
noise_xent = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
  labels=y_input, logits=noise_logits))

total_loss = clean_xent + noise_xent + logits_mse

y_pred = tf.argmax(logits, 1)

correct_prediction = tf.equal(y_pred, y_input)

num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(
  total_loss, global_step=global_step)

# Set up adversary for eval
attack = LinfPGDAttack(model,
                       x_input=x_input,
                       y_input=y_input,
                       epsilon=config['epsilon'],
                       k=config['k'],
                       a=config['a'],
                       random_start=config['random_start'],
                       loss_func=config['loss_func'])

if args.train_alp_pgd_steps > 0:
  print('==== using pgd to generate inputs for logit pairing')
  attack_for_train = LinfPGDAttack(model,
                       x_input=x_input,
                       y_input=y_input,
                       epsilon=config['epsilon'],
                       k=args.train_alp_pgd_steps,
                       a=config['a'] * config['k'] / args.train_alp_pgd_steps,
                       random_start=config['random_start'],
                       loss_func=config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = args.model_dir or config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv', accuracy)
tf.summary.scalar('xent adv', clean_xent / batch_size)
tf.summary.image('images adv train', tf.reshape(x_input + noise_input, [-1, 28, 28, 1]))
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch, y_batch = mnist.train.next_batch(batch_size)

    # Add noise to x_batch
    if args.train_alp_pgd_steps == 0:
      noise_batch = np.random.rand(*x_batch.shape) * NOISE_STD
    else:
      x_adv = attack_for_train.perturb(x_batch, y_batch, sess)
      noise_batch = x_adv - x_batch

    feed_dict = {x_input: x_batch,
                 y_input: y_batch,
                 noise_input: noise_batch}

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(accuracy, feed_dict=feed_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
          num_output_steps * batch_size / training_time))
        training_time = 0.0

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)
      adv_dict = {x_input: x_batch_adv, y_input: y_batch}
      adv_acc = sess.run(accuracy, feed_dict=adv_dict)
      print('    eval on pgd attack: {:.4f}%'.format(adv_acc * 100))
      summary = tf.Summary(value=[
          tf.Summary.Value(tag="acc-on-pgd-attack", simple_value=adv_acc), 
      ])
      summary_writer.add_summary(summary, global_step.eval(sess))
      
      summary = sess.run(merged_summaries, feed_dict=feed_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))


    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=feed_dict)
    end = timer()
    training_time += end - start
