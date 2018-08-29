"""Evaluate a model with attacks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from unrestricted_advex import attacks, load_models, plotting

EVAL_WITH_ATTACKS_DIR = '/tmp/eval_with_attacks'


def run_attack(model, data_iter, attack_fn, max_num_batches=1):
  """
  :param model: Callable batch-input -> batch-probability in [0, 1]
  :param data_iter: NHWC data iterator
  :param attack_fn: Callable (model, x_np, y_np) -> x_adv
  :param max_num_batches: Integer number of batches to stop after
  :return: (logits, labels, x_adv)
  """
  all_labels = []
  all_logits = []
  all_xadv = []
  for i_batch, (x_np, y_np) in enumerate(data_iter):
    assert x_np.shape[-1] == 3 or x_np.shape[-1] == 1, "Data was {}, should be NHWC".format(
      x_np.shape)
    if max_num_batches > 0 and i_batch >= max_num_batches:
      break

    x_adv = attack_fn(model, x_np, y_np)
    logits = model(x_adv)

    all_labels.append(y_np)
    all_logits.append(logits)
    all_xadv.append(x_adv)

  return (np.concatenate(all_logits),
          np.concatenate(all_labels),
          np.concatenate(all_xadv))


def evaluate_tcu_model(model_fn, dataset_iter, attack_list, model_fn_name=None):
  """
  Evaluates a model on a set of attacks and creates plots
  :param model_fn: A function mapping images to logits
  :param dataset_iter: An iterable that returns (batched_images, batched_labels)
  :param attack_list: A list of tuples of (attack_fn, attack_name)
  :param model_fn_name: An optional model name
  """
  for (attack_fn, attack_name) in attack_list:
    print("Executing attack: %s" % attack_name)
    logits, labels, x_adv = run_attack(
      model_fn, dataset_iter, attack_fn, max_num_batches=1)

    preds = (logits[:, 0] < logits[:, 1]).astype(np.int64)
    correct = np.equal(preds, labels).astype(np.float32)
    correct_fracs = np.sum(correct, axis=0) / len(labels)
    print("Fraction correct under %s: %.3f" % (attack_name, correct_fracs))

    results_dir = os.path.join(EVAL_WITH_ATTACKS_DIR, attack_name)
    plotting.save_correct_and_incorrect_adv_images(x_adv, correct, results_dir)

    # Confidence is the value of the larger of the two logits
    confidences = np.max(logits, axis=1)

    # We will plot accuracy at various coverages
    coverages = np.linspace(0.01, .99, 99)

    cov_to_confident_error_idxs = get_coverage_to_confident_error_idxs(
      coverages, preds, confidences, labels, )


    plotting.plot_confident_error_rate(
      coverages, cov_to_confident_error_idxs, len(labels), attack_name, results_dir,
      legend=model_fn_name)


def mnist_valid_check(before, after):
  weight_before = np.sum(np.abs(before), axis=(1, 2, 3))
  weight_after = np.sum(np.abs(after), axis=(1, 2, 3))

  return np.abs(weight_after - weight_before) < weight_before * .1


def get_coverage_to_confident_error_idxs(coverages, preds, confidences, y_true):
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


def evaluate_tcu_mnist_model(model_fn, dataset_iter):
  spsa_attack = (lambda model, x, y:
                 attacks.SpsaAttack(model_fn, (28, 28, 1), epsilon=0.3)
                 .spsa_attack(model, x, y))
  return evaluate_tcu_model(model_fn, dataset_iter, [
    # (attacks.null_attack, 'null_attack'),
    (spsa_attack, 'spsa_attack'),
    (lambda model, x, y: attacks.spatial_attack(
      model, x, y,
      spatial_limits=[10, 10, 10],
      grid_granularity=[10, 10, 10],
      black_border_size=4,
      valid_check=mnist_valid_check),
     'spatial_attack'),
  ])


def evaluate_tcu_images_model(model_fn, dataset_iter, model_fn_name=None):
  spsa_attack = attacks.SpsaAttack(model_fn, (224, 224, 3)).spsa_attack
  return evaluate_tcu_model(model_fn, dataset_iter, [
    (attacks.null_attack, 'null_attack'),
    #    (attacks.spatial_attack, 'spatial_attack'),
    #    (spsa_attack, 'spsa_attack'),
  ], model_fn_name=model_fn_name)


def main():
  model_fn = load_models.get_keras_tcu_model()
  dataset_iter = load_models.get_tcu_dataset_iter(batch_size=32)
  evaluate_tcu_images_model(model_fn, dataset_iter, model_fn_name='Keras TCU model')


if __name__ == '__main__':
  main()
