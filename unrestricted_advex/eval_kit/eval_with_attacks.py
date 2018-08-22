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

from tcu_images import BICYCLE_IDX, BIRD_IDX
from unrestricted_advex.eval_kit import attacks, load_models


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
    assert x_np.shape[-1] == 3 or x_np.shape[-1] == 1, "Data was {}, should be NHWC".format(x_np.shape)
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


def evaluate_tcu_model(model_fn, dataset_iter, attack_list,
                       model_fn_name=None):
  for (attack_fn, attack_name) in attack_list:
    print("Executing attack: %s" % attack_name)
    logits, labels, x_adv = run_attack(
      model_fn, dataset_iter, attack_fn, max_num_batches=1)

    preds = (logits[:, 0] < logits[:, 1]).astype(np.int64)
    correct = np.equal(preds, labels).astype(np.float32)
    correct_fracs = np.sum(correct, axis=0) / len(labels)
    print("Fraction correct under %s: %.3f" % (attack_name, correct_fracs))

    confidences = logits_to_confidences(
      bicycle_logits = logits[:, BICYCLE_IDX],
      bird_logits = logits[:, BIRD_IDX])

    coverages, cov_to_confident_error_idxs = get_coverage_to_confident_error_idxs(
      preds, confidences, labels)

    results_dir = os.path.join(EVAL_WITH_ATTACKS_DIR, attack_name)
    plot_ims(x_adv, correct, results_dir)
    plot_confident_error_rate(
      coverages, cov_to_confident_error_idxs, len(labels), attack_name, results_dir,
      legend=model_fn_name)



def mnist_valid_check(before, after):
  weight_before = np.sum(np.abs(before),axis=(1,2,3))
  weight_after = np.sum(np.abs(after),axis=(1,2,3))

  return np.abs(weight_after-weight_before) < weight_before*.1


def evaluate_mnist_tcu_model(model_fn, dataset_iter):
  spsa_attack = (lambda model, x, y:
                 attacks.SpsaAttack(model_fn, (28, 28, 1), epsilon=0.3)
                 .spsa_attack(model, x, y))
  return evaluate_tcu_model(model_fn, dataset_iter, [
    #(attacks.null_attack, 'null_attack'),
    (spsa_attack, 'spsa_attack'),
    (lambda model, x, y: attacks.spatial_attack(model, x, y,
                                                spatial_limits=[10, 10, 10],
                                                grid_granularity=[10, 10, 10],
                                                black_border_size=4,
                                                valid_check=mnist_valid_check),
     'spatial_attack'),
  ])


def logits_to_confidences(bicycle_logits, bird_logits):
    return np.max(np.vstack([ bicycle_logits, bird_logits]).T, axis=1)


def get_coverage_to_confident_error_idxs(preds, confidences, y_true):
    sorted_confidences = list(sorted(confidences, reverse=True))

    coverages = np.linspace(0.01, .99, 99)
    cov_to_confident_error_idxs = []

    for coverage in coverages:
        threshold = sorted_confidences[int(coverage * len(preds))]
        confident_mask = confidences >= threshold
        confident_error_mask = (y_true != preds) * confident_mask
        confident_error_idx = confident_error_mask.nonzero()[0]

        cov_to_confident_error_idxs.append(confident_error_idx)

    return (coverages, cov_to_confident_error_idxs)


def plot_ims(x_adv, correct, results_dir):
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
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  if image_np.shape[-1] == 3:
    img = Image.fromarray(np.uint8(image_np * 255.), 'RGB')
  else:
    img = Image.fromarray(np.uint8(image_np[:,:,0] * 255.), 'L')
  img.save(filename)


def plot_confident_error_rate(coverages, cov_to_confident_error_idxs, num_examples,
                              attack_name, results_dir, legend=None,
                              title="Risk vs Coverage ({attack_name})"):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(coverages, [float(len(idxs)) / num_examples
                         for idxs in cov_to_confident_error_idxs])
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


def evaluate_images_tcu_model(model_fn, dataset_iter, model_fn_name=None):
  spsa_attack = attacks.SpsaAttack(model_fn, (224, 224, 3)).spsa_attack
  return evaluate_tcu_model(model_fn, dataset_iter, [
   (attacks.null_attack, 'null_attack'),
#    (attacks.spatial_attack, 'spatial_attack'),
#    (spsa_attack, 'spsa_attack'),
  ], model_fn_name=model_fn_name)


def main():
  tcu_dataset_iter = load_models.get_torch_tcu_dataset_iter(
    batch_size=64, shuffle=True)
  model_fn = load_models.get_keras_tcu_model()
  evaluate_images_tcu_model(model_fn, tcu_dataset_iter,
                            model_fn_name='Keras TCU model')


if __name__ == '__main__':
  main()
