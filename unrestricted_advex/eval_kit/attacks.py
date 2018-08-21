from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import product, repeat

import numpy as np
import tensorflow as tf
from cleverhans.attacks import SPSA
from cleverhans.model import Model
from six.moves import xrange


def np_sparse_softmax_cross_entropy_with_logits(
    logits_np, labels_np, graph, sess):
  with graph.as_default():
    labels_tf = tf.placeholder(tf.int32)
    logits_tf = tf.placeholder(tf.float32)
    xent_tf = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels_tf, logits=logits_tf)
    return sess.run(xent_tf, feed_dict={
      labels_tf: labels_np, logits_tf: logits_np})


class CleverhansModelWrapper(Model):
  def __init__(self, model_fn):
    """
    Wrap a callable function that takes a numpy array of shape (N, C, H, W),
    and outputs a numpy vector of length N, with each element in range [0, 1].
    """
    self.nb_classes = 2
    self.model_fn = model_fn

  def fprop(self, x, **kwargs):
    logits_op = tf.py_func(self.model_fn, [x], tf.float32)
    return {'logits': logits_op}


def spsa_attack(model, x_np, y_np, epsilon=(16. / 255)):  # (4. / 255)):
  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=(1,) + x_np.shape[1:])
    y_label = tf.placeholder(tf.int32, shape=(1,))

    cleverhans_model = CleverhansModelWrapper(model)
    attack = SPSA(cleverhans_model)

    x_adv = attack.generate(
      x_input,
      y=y_label,
      epsilon=epsilon,
      num_steps=200,
      early_stop_loss_threshold=-1.,
      spsa_samples=32,
      is_debug=True)

    # Run computation
    all_x_adv_np = []
    with tf.Session() as sess:
      for i in xrange(len(x_np)):
        x_adv_np = sess.run(x_adv, feed_dict={
          x_input: np.expand_dims(x_np[i], axis=0),
          y_label: np.expand_dims(y_np[i], axis=0),
        })
        all_x_adv_np.append(x_adv_np)
    return np.concatenate(all_x_adv_np)


def null_attack(model, x_np, y_np):
  del model, y_np  # unused
  return x_np


def spatial_attack(model, x_np, y_np,
                   spatial_limits=[10, 10, 20],
                   grid_granularity=[5, 5, 21],
                   black_border_size=60):
  attack = SpatialGridAttack(
    model,
    image_shape_hwc=x_np.shape[1:],
    spatial_limits=spatial_limits,
    grid_granularity=grid_granularity,
    black_border_size=black_border_size,
  )
  return attack.perturb_grid(x_input_np=x_np, y_input_np=y_np)


class SpatialGridAttack:
  def __init__(self, model, image_shape_hwc,
               spatial_limits,
               grid_granularity,
               black_border_size,
               ):
    """
    :param model: a callable: batch-input -> batch-probability in [0, 1]
    :param spatial_limits:
    :param grid_granularity:
    """
    self.limits = spatial_limits
    self.granularity = grid_granularity

    # Construct graph for spatial attack
    self.graph = tf.Graph()
    with self.graph.as_default():
      self._x_for_trans = tf.placeholder(tf.float32, shape=[None] + list(image_shape_hwc))
      self._t_for_trans = tf.placeholder(tf.float32, shape=[None, 3])

      x = apply_black_border(
        self._x_for_trans,
        image_height=image_shape_hwc[0],
        image_width=image_shape_hwc[1],
        border_size=black_border_size
      )

      self._tranformed_x_op = apply_transformation(
        x,
        transform=self._t_for_trans,
        image_height=image_shape_hwc[0],
        image_width=image_shape_hwc[1],
      )
      self.session = tf.Session()

    self.model = model
    self.grid_store = []

  def perturb_grid(self, x_input_np, y_input_np):
    n = len(x_input_np)
    grid = product(*list(np.linspace(-l, l, num=g)
                         for l, g in zip(self.limits, self.granularity)))

    worst_x = np.copy(x_input_np)
    max_xent = np.zeros(n)
    all_correct = np.ones(n).astype(bool)

    for horizontal_trans, vertical_trans, rotation in grid:
      trans_np = np.stack(
        repeat([horizontal_trans, vertical_trans, rotation], n))

      # Apply the spatial attack
      with self.graph.as_default():
        x_np = self.session.run(self._tranformed_x_op, feed_dict={
          self._x_for_trans: x_input_np,
          self._t_for_trans: trans_np,

        })
      # See how the model performs on the perturbed input
      logits = self.model(x_np)
      preds = np.argmax(logits, axis=1)

      cur_xent = np_sparse_softmax_cross_entropy_with_logits(
        logits, y_input_np, self.graph, self.session)

      cur_xent = np.asarray(cur_xent)
      cur_correct = np.equal(y_input_np, preds)

      # Select indices to update: we choose the misclassified transformation
      # of maximum xent (or just highest xent if everything else if correct).
      idx = (cur_xent > max_xent) & (cur_correct == all_correct)
      idx = idx | (cur_correct < all_correct)
      max_xent = np.maximum(cur_xent, max_xent)
      all_correct = cur_correct & all_correct

      idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1)

      idx = np.expand_dims(idx, axis=-1)
      idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1, 1, 1)
      worst_x = np.where(idx, x_np, worst_x, )  # shape (bsize, 32, 32, 3)

    return worst_x


def apply_black_border(x, image_height, image_width, border_size):
  x = tf.image.resize_images(x, (image_width - border_size,
                                 image_height - border_size))
  x = tf.pad(x, [[0, 0],
                 [border_size, border_size],
                 [border_size, border_size],
                 [0, 0]], 'CONSTANT')
  return x


def apply_transformation(x, transform, image_height, image_width):
  # Map a transformation onto the input
  trans_x, trans_y, rot = tf.unstack(transform, axis=1)
  rot *= np.pi / 180  # convert degrees to radians

  # Pad the image to prevent two-step rotation / translation
  # resulting in a cropped image
  x = tf.pad(x, [[0, 0], [50, 50], [50, 50], [0, 0]], 'CONSTANT')

  # rotate and translate image
  ones = tf.ones(shape=tf.shape(trans_x))
  zeros = tf.zeros(shape=tf.shape(trans_x))
  trans = tf.stack([ones, zeros, -trans_x,
                    zeros, ones, -trans_y,
                    zeros, zeros], axis=1)
  x = tf.contrib.image.rotate(x, rot, interpolation='BILINEAR')
  x = tf.contrib.image.transform(x, trans, interpolation='BILINEAR')
  print('here shape',x.shape)
  return tf.image.resize_image_with_crop_or_pad(
    x, image_height, image_width)
