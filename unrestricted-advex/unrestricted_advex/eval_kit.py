"""Evaluate a model with attacks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import bird_or_bicyle
import numpy as np
from tqdm import tqdm
from unrestricted_advex import attacks, plotting
from unrestricted_advex.two_class_mnist_dataset import two_class_mnist_dataset

EVAL_WITH_ATTACKS_DIR = '/tmp/eval_with_attacks'


def _validate_logits(logits, batch_size):
  """Validate the model_fn to help the researcher with debugging"""

  desired_logits_shape = (batch_size, 2)
  assert logits.shape == desired_logits_shape, \
    "Your model_fn must return logits with shape %s. Got %s instead." % (
      desired_logits_shape, logits.shape)

  assert logits.dtype == np.float32, \
    "Your model_fn must return logits as np.float32. Got %s instead. \
    Try using logits_np.astype(np.float32)" % logits.dtype


def run_attack(model, data_iter, attack_fn):
  """ Runs an attack on the model_fn for every batch in data_iter and returns the results

  :param model: Callable batch-input -> batch-probability in [0, 1]
  :param data_iter: NHWC data iterator
  :param attack_fn: Callable (model_fn, x_np, y_np) -> x_adv
  :return: (logits, labels, x_adv)
  """
  all_labels = []
  all_logits = []
  all_xadv = []

  # TODO: Add assertion about the model's throughput
  for i_batch, (x_np, y_np) in enumerate(tqdm(data_iter)):
    assert x_np.shape[-1] == 3 or x_np.shape[-1] == 1, "Data was {}, should be NHWC".format(
      x_np.shape)

    x_adv = attack_fn(model, x_np, y_np)
    logits = model(x_adv)

    _validate_logits(logits, batch_size=len(x_np))

    all_labels.append(y_np)
    all_logits.append(logits)
    all_xadv.append(x_adv)

  return (np.concatenate(all_logits),
          np.concatenate(all_labels),
          np.concatenate(all_xadv))


def _evaluate_two_class_unambiguous_model(model_fn, data_iter, attack_list, model_name=None):
  """
  Evaluates a model_fn on a set of attacks and creates plots
  :param model_fn: A function mapping images to logits
  :param dataset_iter: An iterable that returns (batched_images, batched_labels)
  :param attack_list: A list of callable Attacks
  :param model_name: An optional model_fn name
  """
  # Load the whole data_iter into memory because we will iterate through the iterator multiple times
  data_iter = list(data_iter)
  for attack in attack_list:
    print("Executing attack: %s" % attack.name)

    logits, labels, x_adv = run_attack(model_fn, data_iter, attack)

    preds = (logits[:, 0] < logits[:, 1]).astype(np.int64)
    correct = np.equal(preds, labels).astype(np.float32)
    correct_fracs = np.sum(correct, axis=0) / len(labels)
    print("Fraction correct under %s: %.3f" % (attack.name, correct_fracs))

    results_dir = os.path.join(EVAL_WITH_ATTACKS_DIR, attack.name)
    plotting.save_correct_and_incorrect_adv_images(x_adv, correct, results_dir)

    # Confidence is the value of the larger of the two logits
    confidences = np.max(logits, axis=1)

    # We will plot accuracy at various coverages
    coverages = np.linspace(0.01, .99, 99)

    cov_to_confident_error_idxs = _get_coverage_to_confident_error_idxs(
      coverages, preds, confidences, labels, )

    plotting.plot_confident_error_rate(
      coverages, cov_to_confident_error_idxs, len(labels), attack.name, results_dir,
      legend=model_name)


def _get_coverage_to_confident_error_idxs(coverages, preds, confidences, y_true):
  """Returns a list of confident error indices for each coverage"""
  sorted_confidences = list(sorted(confidences, reverse=True))

  cov_to_confident_error_idxs = []

  for coverage in coverages:
    threshold = sorted_confidences[int(coverage * len(preds))]
    confident_mask = confidences >= threshold
    confident_error_mask = (y_true != preds) * confident_mask
    confident_error_idx = confident_error_mask.nonzero()[0]

    cov_to_confident_error_idxs.append(confident_error_idx)

  return cov_to_confident_error_idxs

def evaluate_two_class_mnist_model(model_fn, dataset_iter=None, model_name=None):
  """
  Evaluates a two-class MNIST model_fn on a default set of attacks and creates plots
  :param model_fn: A function mapping images to logits
  :param dataset_iter: An iterable that returns (batched_images, batched_labels)
  :param model_name: An optional model_fn name
  """

  def _mnist_valid_check(before, after):
    weight_before = np.sum(np.abs(before), axis=(1, 2, 3))
    weight_after = np.sum(np.abs(after), axis=(1, 2, 3))
    return np.abs(weight_after - weight_before) < weight_before * .1

  images_2class, labels_2class = two_class_mnist_dataset()

  mnist_label_to_examples = {0: images_2class[0==labels_2class],
                             1: images_2class[1==labels_2class]}

  attack_list = [
    attacks.NullAttack(),
    attacks.SpsaAttack(
      model_fn,
      image_shape_hwc=(28, 28, 1),
      epsilon=0.3),
    attacks.SpatialGridAttack(
      image_shape_hwc=(28, 28, 1),
      spatial_limits=[10, 10, 10],
      grid_granularity=[10, 10, 10],
      black_border_size=4,
      valid_check=_mnist_valid_check),
    attacks.BoundaryAttack(
      model_fn,
      max_l2_distortion=4,
      label_to_examples=mnist_label_to_examples),
  ]

  return _evaluate_two_class_unambiguous_model(
    model_fn, dataset_iter,
    model_name=model_name,
    attack_list=attack_list)


def evaluate_bird_or_bicycle_model(model_fn, dataset_iter=None, model_name=None):
  """
  Evaluates a bird_or_bicycle classifier on a default set of attacks and creates plots
  :param model_fn: A function mapping images to logits
  :param dataset_iter: An iterable that returns (batched_images, batched_labels)
  :param model_name: An optional model_fn name
  """
  if dataset_iter is None:
    dataset_iter = bird_or_bicyle.get_iterator('test')

  attack_list = [
    attacks.NullAttack(),
    attacks.SpsaAttack(
      model_fn,
      image_shape_hwc=(224, 224, 3),
      epsilon=(16. / 255)),
    attacks.SpatialGridAttack(
      image_shape_hwc=(224, 224, 3),
      spatial_limits=[18, 18, 30],
      grid_granularity=[5, 5, 31],
      black_border_size=0),
    attacks.BoundaryAttack(
      model_fn,
      max_l2_distortion=4,
      label_to_example=bird_or_bicycle_label_to_examples),
  ]
  return _evaluate_two_class_unambiguous_model(model_fn, dataset_iter,
                                               model_name=model_name,
                                               attack_list=attack_list)
