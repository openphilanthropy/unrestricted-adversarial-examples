"""Helper functions for running models in a distributed setting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_distribution_strategy(num_gpus, all_reduce_alg=None):
  """Return a DistributionStrategy for running the model.

  Args:
    num_gpus: Number of GPUs to run this model.
    all_reduce_alg: Specify which algorithm to use when performing all-reduce.
      See tf.contrib.distribute.AllReduceCrossTowerOps for available algorithms.
      If None, DistributionStrategy will choose based on device topology.

  Returns:
    tf.contrib.distribute.DistibutionStrategy object.
  """
  if num_gpus == 0:
    return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  else:
    if all_reduce_alg:
      return tf.contrib.distribute.MirroredStrategy(
          num_gpus=num_gpus,
          cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
              all_reduce_alg, num_packs=num_gpus))
    else:
      return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)


def per_device_batch_size(batch_size, num_gpus):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.

  Note that this should eventually be handled by DistributionStrategies
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.

  Args:
    batch_size: Global batch size to be divided among devices. This should be
      equal to num_gpus times the single-GPU batch_size for multi-gpu training.
    num_gpus: How many GPUs are used with DistributionStrategies.

  Returns:
    Batch size per device.

  Raises:
    ValueError: if batch_size is not divisible by number of devices
  """
  if num_gpus <= 1:
    return batch_size

  remainder = batch_size % num_gpus
  if remainder:
    err = ("When running with multiple GPUs, batch size "
           "must be a multiple of the number of available GPUs. Found {} "
           "GPUs with a batch size of {}; try --batch_size={} instead."
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)
  return int(batch_size / num_gpus)
