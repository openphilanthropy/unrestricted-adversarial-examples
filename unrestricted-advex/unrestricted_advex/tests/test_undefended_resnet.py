"""Test examples/undefended_pytorch.

We only run the script for training / evaluation
for one tiny batch to verify that the program can
successfully run without issue. The correctness of
the results are not checked in this auto-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bird_or_bicycle
import numpy as np
import pytest
import tensorflow as tf
import torch
from bird_or_bicycle import CLASS_NAME_TO_IMAGENET_CLASS
from unrestricted_advex import attacks
from unrestricted_advex.eval_kit import evaluate_two_class_unambiguous_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Resnet tests require CUDA")
def test_spatial_speed():
  # Keras isn't being found by travis for some reason
  from tensorflow.keras.applications.resnet50 import preprocess_input

  # Variable scoping lets us use our _graph in our model_fn
  _graph = tf.Graph()
  with _graph.as_default():
    k_model = tf.keras.applications.resnet50.ResNet50(
      include_top=True, weights='imagenet', input_tensor=None,
      input_shape=None, pooling=None, classes=1000)

  def undefended_keras_model_fn(x_np):
    """A normal keras resnet that was pretrained on ImageNet"""
    x_np = preprocess_input(x_np * 255)

    with _graph.as_default():
      prob1000 = k_model.predict_on_batch(x_np) / 10

    # Keras returns softmax-ed probs, we convert them back to logits
    fake_logits1000 = np.log(prob1000)

    bird_max_logit = np.max(
      fake_logits1000[:, CLASS_NAME_TO_IMAGENET_CLASS['bird']], axis=1)
    bicycle_max_logit = np.max(
      fake_logits1000[:, CLASS_NAME_TO_IMAGENET_CLASS['bicycle']], axis=1)

    two_class_logits = np.concatenate((
      bicycle_max_logit[:, None],
      bird_max_logit[:, None]),
      axis=1)
    return two_class_logits

  # Set up standard attack params
  bird_or_bicycle_shape = (224, 224, 3)
  bird_or_bicycle_spatial_limits = [18, 18, 30]
  bird_or_bicycle_black_border_size = 20

  grid_granularity = [5, 5, 31]
  model_fn = undefended_keras_model_fn

  spatial_attack = attacks.FastSpatialGridAttack(
    model_fn,
    image_shape_hwc=bird_or_bicycle_shape,
    spatial_limits=bird_or_bicycle_spatial_limits,
    grid_granularity=grid_granularity,
    black_border_size=bird_or_bicycle_black_border_size,
  )

  ds_size = 32
  spatial_attack._stop_after_n_datapoints = ds_size
  dataset_iter = bird_or_bicycle.get_iterator(
    'train', batch_size=32, verify_dataset=False)
  return evaluate_two_class_unambiguous_model(
    model_fn, dataset_iter,
    model_name='undefended_keras_resnet_test_spatial',
    attack_list=[spatial_attack])



@pytest.mark.skipif(not torch.cuda.is_available(), reason="Resnet tests require CUDA")
def test_common_corruptions():
  # Keras isn't being found by travis for some reason
  from tensorflow.keras.applications.resnet50 import preprocess_input

  # Variable scoping lets us use our _graph in our model_fn
  _graph = tf.Graph()
  with _graph.as_default():
    k_model = tf.keras.applications.resnet50.ResNet50(
      include_top=True, weights='imagenet', input_tensor=None,
      input_shape=None, pooling=None, classes=1000)

  def undefended_keras_model_fn(x_np):
    """A normal keras resnet that was pretrained on ImageNet"""
    x_np = preprocess_input(x_np * 255)

    with _graph.as_default():
      prob1000 = k_model.predict_on_batch(x_np) / 10

    # Keras returns softmax-ed probs, we convert them back to logits
    fake_logits1000 = np.log(prob1000)

    bird_max_logit = np.max(
      fake_logits1000[:, CLASS_NAME_TO_IMAGENET_CLASS['bird']], axis=1)
    bicycle_max_logit = np.max(
      fake_logits1000[:, CLASS_NAME_TO_IMAGENET_CLASS['bicycle']], axis=1)

    two_class_logits = np.concatenate((
      bicycle_max_logit[:, None],
      bird_max_logit[:, None]),
      axis=1)
    return two_class_logits

  # Set up standard attack params
  bird_or_bicycle_shape = (224, 224, 3)
  bird_or_bicycle_spatial_limits = [18, 18, 30]
  bird_or_bicycle_black_border_size = 20

  grid_granularity = [5, 5, 31]
  model_fn = undefended_keras_model_fn

  spatial_attack = attacks.CommonCorruptionsAttack()

  ds_size = 32
  spatial_attack._stop_after_n_datapoints = ds_size
  dataset_iter = bird_or_bicycle.get_iterator(
    'train', batch_size=32, verify_dataset=False)
  return evaluate_two_class_unambiguous_model(
    model_fn, dataset_iter,
    model_name='undefended_keras_resnet_test_common_corruptions',
    attack_list=[spatial_attack])


if __name__ == '__main__':
  test_common_corruptions()
