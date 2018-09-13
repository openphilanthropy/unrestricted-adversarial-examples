"""Evaluate a model with attacks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import bird_or_bicyle
import numpy as np
from terminaltables import AsciiTable
from tqdm import tqdm
from unrestricted_advex import attacks, plotting
from unrestricted_advex.mnist_baselines import mnist_utils

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


def logits_to_preds(logits):
  return (logits[:, 0] < logits[:, 1]).astype(np.int64)


def run_attack(model, data_iter, attack_fn):
  """ Runs an attack on the model_fn for every batch in data_iter and returns the results

  :param model: Callable batch-input -> batch-probability in [0, 1]
  :param data_iter: NHWC data iterator
  :param attack_fn: Callable (model_fn, x_np, y_np) -> x_adv
  :return: (logits, labels, x_adv)
  """
  all_labels = []
  all_logits = []
  all_correct = []
  all_xadv = []

  # TODO: Add assertion about the model's throughput
  for i_batch, (x_np, y_np) in enumerate(tqdm(data_iter)):
    assert x_np.shape[-1] == 3 or x_np.shape[-1] == 1, "Data was {}, should be NHWC".format(
      x_np.shape)

    x_adv = attack_fn(model, x_np, y_np)
    logits = model(x_adv)
    correct = np.equal(logits_to_preds(logits), y_np).astype(np.float32)

    _validate_logits(logits, batch_size=len(x_np))

    all_labels.append(y_np)
    all_logits.append(logits)
    all_correct.append(correct)
    all_xadv.append(x_adv)

  return (np.concatenate(all_logits),
          np.concatenate(all_labels),
          np.concatenate(all_correct),
          np.concatenate(all_xadv))


def evaluate_two_class_unambiguous_model(model_fn, data_iter, attack_list, model_name=None):
  """
  Evaluates a model_fn on a set of attacks and creates plots
  :param model_fn: A function mapping images to logits
  :param dataset_iter: An iterable that returns (batched_images, batched_labels)
  :param attack_list: A list of callable Attacks
  :param model_name: An optional model_fn name

  :return a map from attack_name to accuracy at 80% and 100%
  """
  # Load the whole data_iter into memory because we will iterate through the iterator multiple times
  data_iter = list(data_iter)

  table_data = [
    ['Attack name', 'Acc @ 80% cov', 'Acc @ 100% cov']
  ]
  results = {}
  for attack in attack_list:
    print("Executing attack: %s" % attack.name)

    logits, labels, correct, x_adv = run_attack(model_fn, data_iter, attack)
    correct_fracs = np.sum(correct, axis=0) / len(labels)
    print("Fraction correct under %s: %.3f" % (attack.name, correct_fracs))

    results_dir = os.path.join(EVAL_WITH_ATTACKS_DIR, attack.name)
    plotting.save_correct_and_incorrect_adv_images(x_adv, correct, results_dir)

    # Confidence is the value of the larger of the two logits
    confidences = np.max(logits, axis=1)

    # We will plot accuracy at various coverages
    coverages = np.linspace(0.01, 1.00, 100)

    preds = logits_to_preds(logits)
    cov_to_confident_error_idxs = _get_coverage_to_confident_error_idxs(
      coverages, preds, confidences, labels, )

    plotting.plot_confident_error_rate(
      coverages, cov_to_confident_error_idxs, len(labels), attack.name, results_dir,
      legend=model_name)

    # Add accuracy at 80% and 100% to table and results
    num_errors_at_80 = len(cov_to_confident_error_idxs[80 - 1])
    num_errors_at_100 = len(cov_to_confident_error_idxs[-1])

    acc_at_80 = 1.0 - (float(num_errors_at_80) / len(labels))
    acc_at_100 = 1.0 - (float(num_errors_at_100) / len(labels))

    table_data.append([attack.name, acc_at_80, acc_at_100])
    results[attack.name] = {
      'accuracy@80': acc_at_80,
      'accuracy@100': acc_at_100,
      'cov_to_confident_error_idxs': cov_to_confident_error_idxs
    }

  # Print results
  print(AsciiTable(table_data).table)
  return results


def _get_coverage_to_confident_error_idxs(coverages, preds, confidences, y_true):
  """Returns a list of confident error indices for each coverage"""
  sorted_confidences = list(sorted(confidences, reverse=True))

  cov_to_confident_error_idxs = []
  for coverage in coverages:
    threshold = sorted_confidences[int(coverage * len(preds)) - 1]
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

  images_2class, labels_2class = mnist_utils.two_class_mnist_dataset()

  mnist_label_to_examples = {0: images_2class[0 == labels_2class],
                             1: images_2class[1 == labels_2class]}

  spatial_limits = [10, 10, 10]

  attack_list = [
    attacks.NullAttack(),

    attacks.SpsaAttack(
      model_fn,
      image_shape_hwc=(28, 28, 1),
      epsilon=0.3,
    ),

    attacks.SpatialGridAttack(
      image_shape_hwc=(28, 28, 1),
      spatial_limits=spatial_limits,
      grid_granularity=[10, 10, 10],
      black_border_size=4,
      valid_check=mnist_utils.mnist_valid_check),

    attacks.BoundaryAttack(
      model_fn,
      max_l2_distortion=4,
      label_to_examples=mnist_label_to_examples),
  ]

  return evaluate_two_class_unambiguous_model(
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
    # comment for now, bird_or_bicycle_label_to_example not defined
    # attacks.BoundaryAttack(
    #   model_fn,
    #   max_l2_distortion=4,
    #   label_to_example=bird_or_bicycle_label_to_examples),
  ]
  return evaluate_two_class_unambiguous_model(model_fn, dataset_iter,
                                              model_name=model_name,
                                              attack_list=attack_list)
