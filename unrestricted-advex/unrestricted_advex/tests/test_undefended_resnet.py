"""Test examples/undefended_pytorch.

We only run the script for training / evaluation
for one tiny batch to verify that the program can
successfully run without issue. The correctness of
the results are not checked in this auto-test.
"""
import bird_or_bicycle
import numpy as np
import pytest
import torch
from unrestricted_advex import attacks
from unrestricted_advex.eval_kit import evaluate_two_class_unambiguous_model

from examples.undefended_keras_resnet.main import undefended_keras_model_fn


def my_very_robust_model(images_batch_nhwc):
  """ This function implements a valid unrestricted advex defense that always returns higher
  logits for the second class """
  batch_size = len(images_batch_nhwc)
  logits_np = np.array([[-5.0, 5.0]] * batch_size)
  return logits_np.astype(np.float32)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Resnet tests require CUDA")
def test_spatial_speed():
  """

  Original: 165.14s

  :return:
  """
  bird_or_bicycle_shape = (224, 224, 3)
  bird_or_bicycle_spatial_limits = [18, 18, 30]
  bird_or_bicycle_black_border_size = 25

  grid_granularity = [3, 3, 3]
  model_fn = my_very_robust_model

  spatial_attack = attacks.FastSpatialGridAttack(
    model_fn,
    image_shape_hwc=bird_or_bicycle_shape,
    spatial_limits=bird_or_bicycle_spatial_limits,
    grid_granularity=grid_granularity,
    black_border_size=bird_or_bicycle_black_border_size,
  )

  ds_size = 8
  spatial_attack._stop_after_n_datapoints = ds_size
  dataset_iter = bird_or_bicycle.get_iterator(
    'train', batch_size=ds_size, verify_dataset=False)
  return evaluate_two_class_unambiguous_model(
    model_fn, dataset_iter,
    model_name='undefended_keras_resnet_test,all_positive',
    attack_list=[spatial_attack])


if __name__ == '__main__':
  test_spatial_speed()
