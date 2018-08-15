"""Tests for eval_with_attacks"""
import numpy as np


N, C, H, W = 64, 3, 299, 299


def dummy_data_iter():
  # dummy data is random uniform noise in [0, 1)
  sample = np.random.rand(N, C, H, W)
  means = np.mean(sample, axis=(1, 2, 3))
  yield (sample, means > 0.5)


def dummy_model(batch_nchw):
  # dummy prediction is the mean of each image
  return np.mean(batch_nchw, axis=(1, 2, 3))


def dummy_attack(model, batch_nchw, labels):
  # dummy attack darkens if 1 and lightens if 0
  del model  # unused
  eps = 0.01
  sign = 2 * (labels == 0) - 1  # 1 if 0, -1 if 1
  delta = (eps * sign)[:, None, None, None]
  return np.clip(batch_nchw + delta, 0, 1)
