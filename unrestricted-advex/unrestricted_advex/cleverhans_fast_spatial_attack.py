"""
This code should be pulled into the cleverhans repo when it is
accepted. We have copied it here in the meantime so that people
can use it for the warm-up.

See: https://github.com/tensorflow/cleverhans/pull/623
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from cleverhans import utils
from cleverhans.attacks import Attack
from cleverhans.compat import reduce_max
from cleverhans.compat import reduce_sum
from cleverhans.model import CallableModelWrapper, Model

_logger = utils.create_logger("cleverhans.attacks.tf")

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')


class SpatialTransformationMethod(Attack):
  """
  Spatial transformation attack
  """

  def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
    """
    Create a SpatialTransformationMethod instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """
    if not isinstance(model, Model):
      model = CallableModelWrapper(model, 'probs')

    super(SpatialTransformationMethod, self).__init__(
      model, back, sess, dtypestr)
    self.feedable_kwargs = {
      'n_samples': self.np_dtype,
      'dx_min': self.np_dtype,
      'dx_max': self.np_dtype,
      'n_dxs': self.np_dtype,
      'dy_min': self.np_dtype,
      'dy_max': self.np_dtype,
      'n_dys': self.np_dtype,
      'angle_min': self.np_dtype,
      'angle_max': self.np_dtype,
      'n_angles': self.np_dtype,
      'black_border_size': self.np_dtype,
    }

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.
    :param x: The model's symbolic inputs.
    :param n_samples: (optional) The number of transformations sampled to
                      construct the attack. Set it to None to run
                      full grid attack.
    :param dx_min: (optional float) Minimum translation ratio along x-axis.
    :param dx_max: (optional float) Maximum translation ratio along x-axis.
    :param n_dxs: (optional int) Number of discretized translation ratios
                  along x-axis.
    :param dy_min: (optional float) Minimum translation ratio along y-axis.
    :param dy_max: (optional float) Maximum translation ratio along y-axis.
    :param n_dys: (optional int) Number of discretized translation ratios
                  along y-axis.
    :param angle_min: (optional float) Largest counter-clockwise rotation
                      angle.
    :param angle_max: (optional float) Largest clockwise rotation angle.
    :param n_angles: (optional int) Number of discretized angles.
    :param black_border_size: (optional int) size of the black border in pixels.
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    labels, _ = self.get_or_guess_labels(x, kwargs)

    return spm(
      x,
      self.model,
      y=labels,
      n_samples=self.n_samples,
      dx_min=self.dx_min, dx_max=self.dx_max, n_dxs=self.n_dxs,
      dy_min=self.dy_min, dy_max=self.dy_max, n_dys=self.n_dys,
      angle_min=self.angle_min, angle_max=self.angle_max,
      n_angles=self.n_angles, black_border_size=self.black_border_size)

  def parse_params(self,
                   n_samples=None,
                   dx_min=-0.1,
                   dx_max=0.1,
                   n_dxs=2,
                   dy_min=-0.1,
                   dy_max=0.1,
                   n_dys=2,
                   angle_min=-30,
                   angle_max=30,
                   n_angles=6,
                   black_border_size=0,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.
    """
    self.n_samples = n_samples
    self.dx_min = dx_min
    self.dx_max = dx_max
    self.n_dxs = n_dxs
    self.dy_min = dy_min
    self.dy_max = dy_max
    self.n_dys = n_dys
    self.angle_min = angle_min
    self.angle_max = angle_max
    self.n_angles = n_angles
    self.black_border_size = black_border_size

    if self.dx_min < -1 or self.dy_min < -1 or \
                self.dx_max > 1 or self.dy_max > 1:
      raise ValueError("The value of translation must be bounded "
                       "within [-1, 1]")
    return True


def _apply_black_border(x, border_size):
  orig_height = x.get_shape().as_list()[1]
  orig_width = x.get_shape().as_list()[2]
  x = tf.image.resize_images(x, (orig_width - 2 * border_size,
                                 orig_height - 2 * border_size))

  return tf.pad(x, [[0, 0],
                    [border_size, border_size],
                    [border_size, border_size],
                    [0, 0]], 'CONSTANT')


def _apply_transformation(inputs):
  x, trans = inputs[0], inputs[1]
  dx, dy, angle = trans[0], trans[1], trans[2]
  height = x.get_shape().as_list()[1]
  width = x.get_shape().as_list()[2]

  # Pad the image to prevent two-step rotation / translation from truncating corners
  max_dist_from_center = float(np.max([height, width])) * np.sqrt(2) / 2
  min_edge_from_center = float(np.min([height, width])) / 2
  padding = np.ceil(max_dist_from_center - min_edge_from_center).astype(np.int32)
  x = tf.pad(x, [[0, 0],
                 [padding, padding],
                 [padding, padding],
                 [0, 0]],
             'CONSTANT')

  # Apply rotation
  angle *= np.pi / 180
  x = tf.contrib.image.rotate(x, angle, interpolation='BILINEAR')

  # Apply translation
  dx_in_px = -dx * height
  dy_in_px = -dy * width
  translation = tf.convert_to_tensor([dx_in_px, dy_in_px])
  x = tf.contrib.image.translate(x, translation, interpolation='BILINEAR')
  return tf.image.resize_image_with_crop_or_pad(x, height, width)


def spm(x, model, y=None, n_samples=None, dx_min=-0.1,
        dx_max=0.1, n_dxs=5, dy_min=-0.1, dy_max=0.1, n_dys=5,
        angle_min=-30, angle_max=30, n_angles=31, black_border_size=0):
  """
  TensorFlow implementation of the Spatial Transformation Method.
  :return: a tensor for the adversarial example
  """
  if y is None:
    preds = model.get_probs(x)
    # Using model predictions as ground truth to avoid label leaking
    preds_max = reduce_max(preds, 1, keepdims=True)
    y = tf.to_float(tf.equal(preds, preds_max))
    y = tf.stop_gradient(y)
  y = y / reduce_sum(y, 1, keepdims=True)

  # Define the range of transformations
  dxs = np.linspace(dx_min, dx_max, n_dxs)
  dys = np.linspace(dy_min, dy_max, n_dys)
  angles = np.linspace(angle_min, angle_max, n_angles)

  if n_samples is None:
    import itertools
    transforms = list(itertools.product(*[dxs, dys, angles]))
  else:
    sampled_dxs = np.random.choice(dxs, n_samples)
    sampled_dys = np.random.choice(dys, n_samples)
    sampled_angles = np.random.choice(angles, n_samples)
    transforms = zip(sampled_dxs, sampled_dys, sampled_angles)
  transformed_ims = parallel_apply_transformations(x, transforms, black_border_size)

  def _compute_xent(x):
    preds = model.get_logits(x)
    return tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=y, logits=preds)

  all_xents = tf.map_fn(
    _compute_xent,
    transformed_ims,
    parallel_iterations=1)  # Must be 1 to avoid keras race conditions

  # Return the adv_x with worst accuracy

  # all_xents is n_total_samples x batch_size (SB)
  all_xents = tf.stack(all_xents)  # SB

  # We want the worst case sample, with the largest xent_loss
  worst_sample_idx = tf.argmax(all_xents, axis=0)  # B

  batch_size = tf.shape(x)[0]
  keys = tf.stack([
    tf.range(batch_size, dtype=tf.int32),
    tf.cast(worst_sample_idx, tf.int32)
  ], axis=1)
  transformed_ims_bshwc = tf.einsum('sbhwc->bshwc', transformed_ims)
  after_lookup = tf.gather_nd(transformed_ims_bshwc, keys)  # BHWC
  return after_lookup


def parallel_apply_transformations(x, transforms, black_border_size=0):
  transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
  x = _apply_black_border(x, black_border_size)

  num_transforms = transforms.get_shape().as_list()[0]
  im_shape = x.get_shape().as_list()[1:]

  # Pass a copy of x and a transformation to each iteration of the map_fn callable
  tiled_x = tf.reshape(
    tf.tile(x, [num_transforms, 1, 1, 1]),
    [num_transforms, -1] + im_shape)
  elems = [tiled_x, transforms]
  transformed_ims = tf.map_fn(
    _apply_transformation,
    elems,
    dtype=tf.float32,
    parallel_iterations=1,  # Must be 1 to avoid keras race conditions
  )
  return transformed_ims
