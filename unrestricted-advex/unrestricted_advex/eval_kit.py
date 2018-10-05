"""Evaluate a model with attacks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import bird_or_bicycle
import numpy as np
from terminaltables import AsciiTable
from tqdm import tqdm
from unrestricted_advex import attacks, plotting
from unrestricted_advex.mnist_baselines import mnist_utils


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
  all_image_ids = []

  # TODO: Add assertion about the model's throughput
  for i_batch, (x_np, y_np, image_ids) in enumerate(tqdm(data_iter)):
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
    all_image_ids += list(image_ids)

  return (np.concatenate(all_logits),
          np.concatenate(all_labels),
          np.concatenate(all_correct),
          np.concatenate(all_xadv),
          all_image_ids)


def evaluate_two_class_unambiguous_model(
      model_fn, data_iter, attack_list,
      model_name=None,
      eval_results_dir='/tmp/unrestricted_advex_evals',
):
  """
  Evaluates a model_fn on a set of attacks and creates plots
  :param model_fn: A function mapping images to logits
  :param dataset_iter: An iterable that returns (batched_images, batched_labels, image_ids)
  :param attack_list: A list of callable Attacks
  :param model_name: An optional model_fn name

  :return a map from attack_name to accuracy at 80% and 100%
  """
  if model_name is None:
    model_name = 'unnamed_defense'
  adversarial_images_dir = os.path.join(eval_results_dir, model_name)

  # Load the whole data_iter into memory because we will iterate through the iterator multiple times
  data_iter = list(data_iter)

  table_data = [
    ['Attack name', 'Acc @ 80% cov', 'Acc @ 100% cov']
  ]
  results = {}
  for attack in attack_list:
    print("Executing attack: %s" % attack.name)

    if attack._stop_after_n_datapoints is not None:
      batch_size = len(data_iter[0][0])
      n_batches = int(math.ceil(float(attack._stop_after_n_datapoints) / batch_size))
      attack_data_iter = data_iter[:n_batches]
    else:
      attack_data_iter = data_iter

    logits, labels, correct, x_adv, image_ids = run_attack(model_fn, attack_data_iter, attack)

    results_dir = os.path.join(adversarial_images_dir, attack.name)
    plotting.save_correct_and_incorrect_adv_images(
      x_adv=x_adv,
      correct=correct,
      labels=labels,
      image_ids=image_ids,
      results_dir=results_dir,
    )

    # Confidence is the value of the larger of the two logits
    confidences = np.max(logits, axis=1)

    # We will plot accuracy at various coverages
    desired_coverages = np.linspace(0.01, 1.00, 100)

    preds = logits_to_preds(logits)
    num_covered, confident_error_idxs = _get_num_covered_and_confident_error_idxs(
      desired_coverages, preds, confidences, labels)

    plotting.plot_confident_error_rate(
      desired_coverages, confident_error_idxs, len(labels), attack.name, results_dir,
      legend=model_name)

    # Add accuracy at 80% and 100% to table and results
    num_errors_at_80 = len(confident_error_idxs[80 - 1])
    num_covered_at_80 = num_covered[80 - 1]

    num_errors_at_100 = len(confident_error_idxs[-1])
    num_covered_at_100 = num_covered[-1]

    assert len(logits) == num_covered_at_100
    assert len(logits) == (np.sum(correct, axis=0) + num_errors_at_100)

    acc_at_80 = 1.0 - (float(num_errors_at_80) / num_covered_at_80)
    acc_at_100 = 1.0 - (float(num_errors_at_100) / num_covered_at_100)

    table_data.append([attack.name, acc_at_80, acc_at_100])
    results[attack.name] = {
      'accuracy@80': acc_at_80,
      'accuracy@100': acc_at_100,
      'confident_error_idxs@80': confident_error_idxs[80 - 1],
      'confident_error_idxs@100': confident_error_idxs[-1],
    }
    print("[%s] acc@80%% = %.3f" % (attack.name, acc_at_80))

  # Print results
  print(AsciiTable(table_data).table)
  print("To visually inspect the adversarial examples "
        "that your model correctly and incorrectly classified, "
        "view them in: %s " % adversarial_images_dir)
  return results


def _get_num_covered_and_confident_error_idxs(desired_coverages, preds, confidences, y_true):
  """Returns the number of covered samples and a list of confident error indices for each coverage"""
  sorted_confidences = list(sorted(confidences, reverse=True))

  confident_error_idxs = []
  num_covered = []
  for coverage in desired_coverages:
    threshold = sorted_confidences[int(coverage * len(preds)) - 1]
    confident_mask = confidences >= threshold
    confident_error_mask = (y_true != preds) * confident_mask
    confident_error_idx = confident_error_mask.nonzero()[0]

    confident_error_idxs.append(confident_error_idx)
    num_covered.append(np.sum(confident_mask))

  return num_covered, confident_error_idxs


def evaluate_two_class_mnist_model(model_fn, dataset_iter=None, model_name=None):
  """
  Evaluates a two-class MNIST model_fn on a default set of attacks and creates plots
  :param model_fn: A function mapping images to logits
  :param dataset_iter: An iterable that returns (batched_images, batched_labels)
  :param model_name: An optional model_fn name
  """

  if dataset_iter is None:
    dataset_iter = mnist_utils.get_two_class_iterator('test', num_datapoints=128, batch_size=128)

  mnist_spatial_limits = [10, 10, 10]
  mnist_shape = (28, 28, 1)
  mnist_black_border_size = 0.15

  attack_list = [
    attacks.CleanData(),

    attacks.SpatialGridAttack(
      grid_granularity=[5, 5, 11],
      valid_check=mnist_utils.mnist_valid_check,
      spatial_limits=mnist_spatial_limits,
      black_border_size=mnist_black_border_size,
      image_shape_hwc=mnist_shape,
    ),

    attacks.SpsaWithRandomSpatialAttack(
      model_fn,
      epsilon=0.3,
      valid_check=mnist_utils.mnist_valid_check,
      spatial_limits=mnist_spatial_limits,
      black_border_size=mnist_black_border_size,
      image_shape_hwc=mnist_shape,
      num_steps=2000,
    ),

    attacks.BoundaryWithRandomSpatialAttack(
      model_fn,
      max_l2_distortion=4,
      valid_check=mnist_utils.mnist_valid_check,
      label_to_examples=_get_mnist_labels_to_examples(),
      spatial_limits=mnist_spatial_limits,
      black_border_size=mnist_black_border_size,
      image_shape_hwc=mnist_shape,
    )
  ]

  return evaluate_two_class_unambiguous_model(
    model_fn, dataset_iter,
    model_name=model_name,
    attack_list=attack_list)


def _get_mnist_labels_to_examples():
  images_2class, labels_2class = mnist_utils._two_class_mnist_dataset()
  return {0: images_2class[0 == labels_2class],
          1: images_2class[1 == labels_2class]}


def _get_bird_or_bicycle_label_to_examples():
  dataset_iter = bird_or_bicycle.get_iterator('test')
  label_to_examples = {0: [], 1: []}

  for x_np, y_np, _ in dataset_iter:
    for x, label in zip(x_np, y_np):
      label_to_examples[label].append(x)

  return label_to_examples


def evaluate_bird_or_bicycle_model(model_fn, dataset_iter=None, model_name=None):
  """
  Evaluates a bird_or_bicycle classifier on a default set of attacks and creates plots
  :param model_fn: A function mapping images to logits
  :param dataset_iter: An iterable that returns (batched_images, batched_labels, image_ids)
  :param model_name: An optional model_fn name
  """
  if dataset_iter is None:
    dataset_iter = bird_or_bicycle.get_iterator('test', batch_size=4)

  bird_or_bicycle_shape = (224, 224, 3)
  bird_or_bicycle_spatial_limits = [18, 18, 30]
  bird_or_bicycle_black_border_size = 0.15

  attack_list = [
    attacks.CleanData(),

    attacks.SimpleSpatialAttack(
      spatial_limits=bird_or_bicycle_spatial_limits,
      grid_granularity=[5, 5, 31],
      black_border_size=bird_or_bicycle_black_border_size,
    ),

    attacks.SpsaWithRandomSpatialAttack(
      model_fn,
      image_shape_hwc=bird_or_bicycle_shape,
      spatial_limits=bird_or_bicycle_spatial_limits,
      black_border_size=0,
      epsilon=(16. / 255),
      num_steps=200,
    ),
  ]

  boundary_attack = attacks.BoundaryWithRandomSpatialAttack(
    model_fn,
    max_l2_distortion=10,
    label_to_examples=_get_bird_or_bicycle_label_to_examples(),
    spatial_limits=bird_or_bicycle_spatial_limits,
    black_border_size=bird_or_bicycle_black_border_size,
    image_shape_hwc=bird_or_bicycle_shape,
  )

  # We limit the boundary attack to the first datapoints to speed up eval
  boundary_attack._stop_after_n_datapoints = 100
  attack_list.append(boundary_attack)

  return evaluate_two_class_unambiguous_model(
    model_fn, dataset_iter,
    model_name=model_name,
    attack_list=attack_list)
