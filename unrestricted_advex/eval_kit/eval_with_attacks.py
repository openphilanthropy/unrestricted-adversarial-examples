"""Evaluate a model with attacks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tcu_images
import tensorflow as tf
import torch
import torchvision
from tcu_images import CLASS_NAME_TO_IMAGENET_CLASS, BICYCLE_IDX, BIRD_IDX
from tensorflow.keras.applications.resnet50 import preprocess_input

from unrestricted_advex.eval_kit import attacks

EVAL_WITH_ATTACKS_DIR = '/tmp/eval_with_attacks'

def run_attack(model, data_iter, attack_fn, max_num_batches=1):
  """
  :param model: Model should output a
  :param data_iter: NHWC data
  :param attack:
  :param max_num_batches:
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


def save_image_to_png(image_np, filename):
  from PIL import Image
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  if image_np.shape[-1] == 3:
    img = Image.fromarray(np.uint8(image_np * 255.), 'RGB')
  else:
    img = Image.fromarray(np.uint8(image_np[:,:,0] * 255.), 'L')
  img.save(filename)


def get_torch_tcu_dataset_iter(batch_size, shuffle=True):
  data_dir = tcu_images.get_dataset('train')

  train_dataset = torchvision.datasets.ImageFolder(
    data_dir,
    torchvision.transforms.Compose([
      torchvision.transforms.Resize(224),
      torchvision.transforms.ToTensor(),
      lambda x: torch.einsum('chw->hwc', [x]),
    ]))

  data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=shuffle)

  assert train_dataset.class_to_idx['bicycle'] == BICYCLE_IDX
  assert train_dataset.class_to_idx['bird'] == BIRD_IDX

  dataset_iter = [(x.numpy(), y.numpy()) for (x, y) in iter(data_loader)]
  return dataset_iter


def get_torch_tcu_model():
  print("WARNING: Torch model currently only gets 50% top1 accuracy and may have a preprocessing issue")

  pytorch_model = torchvision.models.resnet50(pretrained=True)
  pytorch_model = pytorch_model.cuda()
  pytorch_model.eval()  # switch to eval mode

  def model_fn(x_np):
    with torch.no_grad():
      x = torch.from_numpy(x_np).cuda()
      x = torch.einsum('bhwc->bchw', [x])
      logits1000 = pytorch_model(x)

      bird_max_logit, _ = torch.max(
        logits1000[:, CLASS_NAME_TO_IMAGENET_CLASS['bird']], dim=1)
      bicycle_max_logit, _ = torch.max(
        logits1000[:, CLASS_NAME_TO_IMAGENET_CLASS['bicycle']], dim=1)
      logits = torch.cat((bicycle_max_logit[:, None],
                          bird_max_logit[:, None]), dim=1)
      return logits.cpu().numpy()

  return model_fn


def get_keras_tcu_model():
  tf.keras.backend.set_image_data_format('channels_last')
  _graph = tf.Graph()
  with _graph.as_default():
    k_model = tf.keras.applications.resnet50.ResNet50(
      include_top=True, weights='imagenet', input_tensor=None,
      input_shape=None, pooling=None, classes=1000)

  def model_wrapper(x_np):
    # it seems keras pre-trained model directly output softmax-ed probs
    x_np = preprocess_input(x_np * 255)

    with _graph.as_default():
      prob1000 = k_model.predict_on_batch(x_np) / 10

    fake_logits1000 = np.log(prob1000)

    bird_max_logit = np.max(
      fake_logits1000[:, CLASS_NAME_TO_IMAGENET_CLASS['bird']], axis=1)
    bicycle_max_logit = np.max(
      fake_logits1000[:, CLASS_NAME_TO_IMAGENET_CLASS['bicycle']], axis=1)
    logits = np.concatenate((bicycle_max_logit[:, None],
                             bird_max_logit[:, None]), axis=1)
    return logits

  return model_wrapper


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


def show(img):
  remap = " .*#"+"#"*100
  img = (img.flatten())*3
  print("START")
  for i in range(28):
    print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def mnist_valid_check(before, after):
  weight_before = np.sum(np.abs(before),axis=(1,2,3))
  weight_after = np.sum(np.abs(after),axis=(1,2,3))

  return np.abs(weight_after-weight_before) < weight_before*.1


def evaluate_mnist_tcu_model(model_fn, dataset_iter):
  return evaluate_tcu_model(model_fn, dataset_iter, [
    #(attacks.null_attack, 'null_attack'),
    (lambda model, x, y: attacks.spsa_attack(model, x, y,
                                             epsilon=0.3),
     'spsa_attack'),
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
  # (attacks.null_attack, 'null_attack'),
#    (attacks.spatial_attack, 'spatial_attack'),
    (spsa_attack, 'spsa_attack'),
  ], model_fn_name=model_fn_name)


def main():
  tcu_dataset_iter = get_torch_tcu_dataset_iter(batch_size=64, shuffle=True)
  model_fn = get_keras_tcu_model()
  evaluate_images_tcu_model(model_fn, tcu_dataset_iter,
                            model_fn_name='Keras TCU model')


if __name__ == '__main__':
  main()
