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
import pytest
import torch
from unrestricted_advex import attacks
from unrestricted_advex.eval_kit import evaluate_two_class_unambiguous_model

from examples.undefended_keras_resnet.main import undefended_keras_model_fn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Resnet tests require CUDA")
def test_spatial_speed():
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


if __name__ == '__main__':
  test_spatial_speed()
