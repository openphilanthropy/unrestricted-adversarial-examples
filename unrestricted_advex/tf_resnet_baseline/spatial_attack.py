from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import product, repeat

import numpy as np
import tensorflow as tf
from absl import flags
from tensorflow.python.estimator.model_fn import ModeKeys
from unrestricted_advex.tf_resnet_baseline.imagenet_main import IM_SHAPE
from unrestricted_advex.tf_resnet_baseline.imagenet_main import model_fn
from unrestricted_advex.tf_resnet_baseline.official_imagenet_input_pipeline import _DEFAULT_IMAGE_SIZE
from unrestricted_advex.tf_resnet_baseline.utils import flag_definitions

FLAGS = flags.FLAGS

pad_mode = 'CONSTANT'


class SpatialGridAttack:
  def __init__(self, model,
               spatial_limits=[3, 3, 30],
               grid_granularity=[5, 5, 31],
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
      self._x_for_trans = tf.placeholder(tf.float32, shape=[None] + IM_SHAPE)
      self._t_for_trans = tf.placeholder(tf.float32, shape=[None, 3])
      self._tranformed_x_op = apply_transformation(
        self._x_for_trans, self._t_for_trans)
    self.session = tf.Session()

    self.model = model
    self.grid_store = []

  def perturb(self, x_nat, y_sparse, sess):
    # Grid attack
    return self.perturb_grid(x_nat, y_sparse, sess)

  def perturb_grid(self, x_input_np, y_input_np, sess):
    n = len(x_input_np)
    grid = product(*list(np.linspace(-l, l, num=g)
                         for l, g in zip(self.limits, self.granularity)))

    worst_x = np.copy(x_input_np)
    worst_t = np.zeros([n, 3])
    max_xent = np.zeros(n)
    all_correct = np.ones(n).astype(bool)

    for horizontal_trans, vertical_trans, rotation in grid:
      trans_np = np.stack(
        repeat([horizontal_trans, vertical_trans, rotation], n))

      # Apply the spatial attack
      x_np = self.session.run(self._tranformed_x_op, feed_dict={
        self._x_for_trans: x_input_np,
        self._t_for_trans: trans_np,

      })

      # See how the model performs on the perturbed input
      cur_xent, predictions = self.model(x_np, y_input_np)

      cur_xent = np.asarray(cur_xent)
      cur_correct = np.equal(y_input_np, predictions)

      # Select indices to update: we choose the misclassified transformation
      # of maximum xent (or just highest xent if everything else if correct).
      idx = (cur_xent > max_xent) & (cur_correct == all_correct)
      idx = idx | (cur_correct < all_correct)
      max_xent = np.maximum(cur_xent, max_xent)
      all_correct = cur_correct & all_correct

      idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1)
      worst_t = np.where(idx, trans_np, worst_t)  # shape (bsize, 3)

      idx = np.expand_dims(idx, axis=-1)
      idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1, 1, 1)
      worst_x = np.where(idx, x_np, worst_x, )  # shape (bsize, 32, 32, 3)

    return worst_x, worst_t


def apply_transformation(x, transform):
  # Map a transformation onto the input
  trans_x, trans_y, rot = tf.unstack(transform, axis=1)
  rot *= np.pi / 180  # convert degrees to radians

  x = tf.pad(x, [[0, 0], [16, 16], [16, 16], [0, 0]], pad_mode)

  # rotate and translate image
  ones = tf.ones(shape=tf.shape(trans_x))
  zeros = tf.zeros(shape=tf.shape(trans_x))
  trans = tf.stack([ones, zeros, -trans_x,
                    zeros, ones, -trans_y,
                    zeros, zeros], axis=1)
  x = tf.contrib.image.rotate(x, rot, interpolation='BILINEAR')
  x = tf.contrib.image.transform(x, trans, interpolation='BILINEAR')
  return tf.image.resize_image_with_crop_or_pad(
    x, _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE)


class EvalModeAttackableModel:
  def __init__(self):
    x_input = tf.placeholder(tf.float32, shape=[None] + IM_SHAPE)
    y_input = tf.placeholder(tf.int32, shape=[None])

    # Create the model in eval mode
    estimator_spec = model_fn(
      x_input, y_input, ModeKeys.EVAL,
      params={
        'resnet_size': int(FLAGS.resnet_size),
        'data_format': FLAGS.data_format,
        'batch_size': FLAGS.batch_size,
        'resnet_version': int(FLAGS.resnet_version),
        'loss_scale': flag_definitions.get_loss_scale(FLAGS),
        'dtype': flag_definitions.get_tf_dtype(FLAGS),
        'use_pgd_attack': False,
      })
    outputs = estimator_spec.predictions

    # Expose inputs and outputs
    self.x_input = x_input
    self.y_input = y_input

    self.cross_entropy = outputs['cross_entropy']
    self.logits = outputs['logits']
    self.probabilities = outputs['probabilities']
    self.predictions = outputs['predictions']
