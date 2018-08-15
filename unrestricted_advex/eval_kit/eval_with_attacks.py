"""Evaluate a model with attacks."""

from six.moves import xrange
from cleverhans.model import Model
from cleverhans.attacks import SPSA
import numpy as np
import tensorflow as tf


def make_cleverhans_model(model):
    return None


def spsa_attack(model, batch_nchw, labels, epsilon=(4. / 255)):
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=(1,) + batch_nchw.shape[1:])
        y_label = tf.placeholder(tf.int32, shape=(1,))

        cleverhans_model = make_cleverhans_model(model)

        attack = SPSA(cleverhans_model)
        x_adv = attack.generate(
            x_input, y=y_label, epsilon=epsilon, num_steps=30,
            early_stop_loss_threshold=-1., batch_size=batch_nchw.shape[0],
            spsa_iters=16, is_debug=True)

        # Run computation
        with tf.Session() as sess:
            for i in xrange(batch_nchw.shape[0]):
                x_adv_np = sess.run(x_adv, feed_dict={
                    x_input: np.expand_dims(batch_nchw[i], axis=0),
                    y_label: np.expand_dims(labels[i], axis=0),
                })
        return x_adv_np


def evaluate(model, data_iter, attacks=None, max_num_batches=1):
  if attacks is None:
    attacks = ['null']  # a single null attack

  all_labels = []
  all_preds = [[] for _ in attacks]

  for i_batch, (x_np, y_np) in enumerate(data_iter()):
    if max_num_batches > 0 and i_batch >= max_num_batches:
      break

    for attack_f, preds_container in zip(attacks, all_preds):
      if attack_f == 'null':
        x_adv = x_np
      else:
        x_adv = attack_f(model, x_np, y_np)

      y_pred = model(x_adv)
      y_pred = np.clip(y_pred, 0, 1)  # force into [0, 1]

      all_labels.append(y_np)
      preds_container.append(y_pred)

  all_labels = np.concatenate(all_labels)
  all_preds = [np.concatenate(x) for x in all_preds]

  return all_labels, all_preds
