""" Tests for distribution util functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order
from unrestricted_advex.tf_official_resnet_baseline.utils.misc import distribution_utils


class GetDistributionStrategyTest(tf.test.TestCase):
  """Tests for get_distribution_strategy."""
  def test_one_device_strategy_cpu(self):
    ds = distribution_utils.get_distribution_strategy(0)
    self.assertTrue(ds.is_single_tower)
    self.assertEquals(ds.num_towers, 1)
    self.assertEquals(len(ds.worker_devices), 1)
    self.assertIn('CPU', ds.worker_devices[0])

  def test_one_device_strategy_gpu(self):
    ds = distribution_utils.get_distribution_strategy(1)
    self.assertTrue(ds.is_single_tower)
    self.assertEquals(ds.num_towers, 1)
    self.assertEquals(len(ds.worker_devices), 1)
    self.assertIn('GPU', ds.worker_devices[0])

  def test_mirrored_strategy(self):
    ds = distribution_utils.get_distribution_strategy(5)
    self.assertFalse(ds.is_single_tower)
    self.assertEquals(ds.num_towers, 5)
    self.assertEquals(len(ds.worker_devices), 5)
    for device in ds.worker_devices:
      self.assertIn('GPU', device)


class PerDeviceBatchSizeTest(tf.test.TestCase):
  """Tests for per_device_batch_size."""

  def test_batch_size(self):
    self.assertEquals(
        distribution_utils.per_device_batch_size(147, num_gpus=0), 147)
    self.assertEquals(
        distribution_utils.per_device_batch_size(147, num_gpus=1), 147)
    self.assertEquals(
        distribution_utils.per_device_batch_size(147, num_gpus=7), 21)

  def test_batch_size_with_remainder(self):
    with self.assertRaises(ValueError):
        distribution_utils.per_device_batch_size(147, num_gpus=5)


if __name__ == "__main__":
  tf.test.main()
