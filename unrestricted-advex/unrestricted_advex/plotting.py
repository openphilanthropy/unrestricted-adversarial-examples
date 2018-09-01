"""Evaluate a model with attacks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



def save_correct_and_incorrect_adv_images(x_adv, correct, results_dir):
  correct_dir = os.path.join(results_dir, 'correct_images')
  shutil.rmtree(correct_dir, ignore_errors=True)

  incorrect_dir = os.path.join(results_dir, 'incorrect_images')
  shutil.rmtree(incorrect_dir, ignore_errors=True)

  for i, image_np in enumerate(x_adv):
    if correct[i]:
      save_dir = correct_dir
    else:
      save_dir = incorrect_dir
    save_image_to_png(image_np, os.path.join(save_dir, "adv_image_%s.png" % i))


def save_image_to_png(image_np, filename):
  from PIL import Image
  dirname = os.path.dirname(filename)
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  if image_np.shape[-1] == 3:
    img = Image.fromarray(np.uint8(image_np * 255.), 'RGB')
  else:
    img = Image.fromarray(np.uint8(image_np[:, :, 0] * 255.), 'L')
  img.save(filename)


def plot_confident_error_rate(coverages, cov_to_confident_error_idxs, num_examples,
                              attack_name, results_dir, legend=None,
                              title="Risk vs Coverage ({attack_name})"):
  """Plot the confident error rate (risk on covered data vs coverage)"""
  fig = plt.figure()
  ax = fig.add_subplot(111)

  cov_to_num_covered_examples = [
    num_examples * coverage for coverage in coverages
  ]

  risk_on_covered_data = [
    float(len(error_idxs)) / num_covered_examples
    for error_idxs, num_covered_examples in
    zip(cov_to_confident_error_idxs, cov_to_num_covered_examples)]

  plt.plot(coverages, risk_on_covered_data)
  plt.title(title.format(attack_name=attack_name))
  plt.ylabel("Risk \n (error rate on covered data)")
  plt.xlabel("Coverage \n (fraction of points not abstained on)")

  if legend:
    ax.legend([legend], loc='best', fontsize=15)

  for item in ([ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(15)
  for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
  ax.title.set_fontsize(15)

  plt.tight_layout()
  plt.savefig(os.path.join(results_dir,
                           "confident_error_rate_{}.png".format(attack_name)))
