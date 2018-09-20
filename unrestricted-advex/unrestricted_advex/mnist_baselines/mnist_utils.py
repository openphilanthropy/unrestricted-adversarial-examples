import itertools
import math
import os

import numpy as np
import tensorflow as tf
from cleverhans.attacks import MadryEtAl
from tensorflow.examples.tutorials.mnist import input_data
from unrestricted_advex.mnist_baselines import mnist_convnet

NUM_CLASSES = 10


def mnist_dataset(one_hot):
  return input_data.read_data_sets('MNIST_data', one_hot=one_hot)


def labels_to_one_hot(labels):
  one_hot = np.zeros((len(labels), NUM_CLASSES))
  one_hot[range(len(one_hot)), labels] = 1
  return one_hot


def one_hot_to_labels(one_hot):
  _, labels = np.where(one_hot)
  return labels


def mnist_valid_check(before, after):
  weight_before = np.sum(np.abs(before), axis=(1, 2, 3))
  weight_after = np.sum(np.abs(after), axis=(1, 2, 3))
  return np.abs(weight_after - weight_before) < weight_before * .1


class TwoClassWrapper(object):
  def __init__(self, mnist_model, classes=(6, 7)):
    self.mnist_model = mnist_model
    self.classes = classes

  def __call__(self, xs):
    logits = self.mnist_model(xs)
    return tf.stack([logits[:, self.classes[0]],
                     logits[:, self.classes[1]]], axis=1)


CLASS1 = 6
CLASS2 = 7


def _two_class_mnist_dataset(one_hot=False):
  mnist = mnist_dataset(one_hot=one_hot)
  which = (mnist.test.labels == CLASS1) | (mnist.test.labels == CLASS2)
  images_2class = mnist.test.images[which].astype(np.float32)

  labels_2class = mnist.test.labels[which] == CLASS2
  images_2class = images_2class.reshape((-1, 28, 28, 1))
  return images_2class, labels_2class


def two_class_iter(images_10class, labels_10class, num_datapoints, batch_size,
                   class1=7, class2=6, label_scheme='boolean',
                   cycle=False):
  """Filter MNIST to only two classes (e.g. sixes and sevens)"""

  # Use the idxs as our image_id so that we can track them down
  image_ids_2class, = np.where((labels_10class == class1) | (labels_10class == class2))

  # Filter to our desired classes
  images_2class = images_10class[image_ids_2class].astype(np.float32)
  labels_2class = labels_10class[image_ids_2class]

  num_batches = int(math.ceil(num_datapoints / batch_size))

  batch_idxs = range(num_batches)
  if cycle:
    batch_idxs = itertools.cycle(batch_idxs)
  for i in batch_idxs:
    image_batch = images_2class[i:i + batch_size].reshape((batch_size, 28, 28, 1))
    image_ids_batch = image_ids_2class[i:i + batch_size]
    if label_scheme == 'boolean':
      labels_batch = labels_2class[i:i + batch_size] == class1
    elif label_scheme == 'one_hot':
      labels_batch = labels_to_one_hot(labels_2class[i:i + batch_size])
    elif label_scheme == 'labels':
      labels_batch = labels_2class[i:i + batch_size]
    else:
      raise NotImplementedError(
        'Unrecognized label scheme {}'.format(label_scheme))

    yield image_batch, labels_batch, list(image_ids_batch)


def np_two_class_mnist_model(model_dir):
  with tf.Graph().as_default():
    x_input = tf.placeholder(tf.float32, (None, 28, 28, 1))
    sess = tf.Session()
    model = mnist_convnet.Model(model_dir, sess)
    two_class_model = TwoClassWrapper(model)
    logits = two_class_model(x_input)

    def np_model(x):
      return sess.run(logits, {x_input: x})

    return np_model


def train_mnist(model_dir, next_batch_fn, total_batches, train_mode,
                save_every=1000, print_every=100):
  x_input = tf.placeholder(tf.float32, (None, 28, 28, 1))
  y_input = tf.placeholder(tf.float32, [None, 10])

  model = mnist_convnet.Model()
  logits = model(x_input)

  loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_input,
                                                 logits=logits)
  loss = tf.reduce_mean(loss)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1),
                                             tf.argmax(y_input, axis=1)),
                                    dtype=tf.float32))

  global_step = tf.contrib.framework.get_or_create_global_step()
  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss,
                                                     global_step=global_step)

  saver = tf.train.Saver(max_to_keep=3)
  a = tf.summary.scalar('accuracy adv train', accuracy)
  b = tf.summary.scalar('xent adv train', loss)
  c = tf.summary.image('images adv train', x_input)
  adv_summaries = tf.summary.merge([a, b, c])

  a = tf.summary.scalar('accuracy nat train', accuracy)
  b = tf.summary.scalar('xent nat train', loss)
  c = tf.summary.image('images nat train', x_input)
  nat_summaries = tf.summary.merge([a, b, c])

  with tf.Session() as sess:
    attack = MadryEtAl(model, sess=sess)

    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    sess.run(tf.global_variables_initializer())

    for batch_num in range(total_batches):
      x_batch, y_batch = next_batch_fn()
      x_batch = np.reshape(x_batch, (-1, 28, 28, 1))

      if train_mode == "adversarial" and batch_num > 1000:
        x_batch_adv = attack.generate_np(x_batch, y=y_batch, eps=.3,
                                         nb_iter=40, eps_iter=.01,
                                         rand_init=True,
                                         clip_min=0, clip_max=1)

      else:
        x_batch_adv = x_batch

      nat_dict = {x_input: x_batch,
                  y_input: y_batch}

      adv_dict = {x_input: x_batch_adv,
                  y_input: y_batch}

      if batch_num % print_every == 0:
        a, l, s = sess.run((accuracy, loss, nat_summaries), nat_dict)
        summary_writer.add_summary(s, sess.run(global_step))
        print(batch_num, "Clean accuracy", a, "loss", l)
        if train_mode == "adversarial":
          a, l, s = sess.run((accuracy, loss, adv_summaries), adv_dict)
          summary_writer.add_summary(s, sess.run(global_step))
          print(batch_num, "Adv accuracy", a, "loss", l)

      if batch_num % save_every == 0:
        saver.save(sess, os.path.join(model_dir, "checkpoint"),
                   global_step=global_step)

      sess.run(train_step, nat_dict)
      sess.run(train_step, adv_dict)
