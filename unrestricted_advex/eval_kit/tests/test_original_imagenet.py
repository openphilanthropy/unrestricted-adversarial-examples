"""Evaluate a model with attacks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import torch
import torchvision
from tensorflow.keras.applications.resnet50 import preprocess_input

from unrestricted_advex.eval_kit import attacks


def run_attack(model, data_iter, attack_fn, max_num_batches=1, save_image_dir=None):
  """
  :param model: Model should output a
  :param data_iter: NHWC data
  :param attack:
  :param max_num_batches:
  :return: (logits, labels)
  """
  all_labels = []
  all_logits = []
  for i_batch, (x_np, y_np) in enumerate(data_iter):
    assert x_np.shape[-1] == 3, "Data was {}, should be NHWC".format(x_np.shape)
    if max_num_batches > 0 and i_batch >= max_num_batches:
      break

    x_adv = attack_fn(model, x_np, y_np)
    logits = model(x_adv)
    all_labels.append(y_np)
    all_logits.append(logits)

  if save_image_dir:
    for i, image_np in enumerate(x_adv):
      save_image_to_png(image_np, os.path.join(save_image_dir, "adv_image_%s.png" % i))

  return np.concatenate(all_logits), np.concatenate(all_labels)


def save_image_to_png(image_np, filename):
  from PIL import Image
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  img = Image.fromarray(np.uint8(image_np * 255.), 'RGB')
  img.save(filename)


def get_torch_imagenet_dataset_iter(batch_size):
  data_dir = '/root/datasets/unpacked_imagenet_pytorch/val'

  train_dataset = torchvision.datasets.ImageFolder(
    data_dir,
    torchvision.transforms.Compose([
      torchvision.transforms.Resize(256),
      torchvision.transforms.CenterCrop(224),
      torchvision.transforms.ToTensor(),
      lambda x: torch.einsum('chw->hwc', [x]),
    ]))

  data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True)
  dataset_iter = ((x.numpy(), y.numpy()) for (x, y) in iter(data_loader))
  return dataset_iter


def get_keras_imagenet_model():
  tf.keras.backend.set_image_data_format('channels_last')

  k_model = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000)

  def model_wrapper(x_np):
    # it seems keras pre-trained model directly output softmax-ed probs
    x_np = preprocess_input(x_np * 255)
    prob1000 = k_model.predict_on_batch(x_np) / 10
    fake_logits1000 = np.log(prob1000)
    return fake_logits1000

  return model_wrapper


def get_torch_imagenet_model():
  print("WARNING: Torch model currently only gets 50% top1 accuracy and may have a preprocessing issue")
  pytorch_model = torchvision.models.resnet50(pretrained=True)
  pytorch_model = pytorch_model.cuda()
  pytorch_model.eval()  # switch to eval mode

  def model_fn(x_np):
    with torch.no_grad():
      x = torch.from_numpy(x_np).cuda()
      x = torch.einsum('bhwc->bchw', [x])
      logits = pytorch_model(x)
      return logits.cpu().numpy()

  return model_fn


def evaluate_imagenet_model(model_fn, dataset_iter):
  for (attack_fn, attack_name) in [
    (attacks.null_attack, 'null_attack'),
    (attacks.spatial_attack, 'spatial_attack'),
    # (attacks.spsa_attack, 'spsa_attack'),
  ]:
    print("Executing attack: %s" % attack_name)
    logits, labels = run_attack(
      model_fn, dataset_iter, attack_fn, max_num_batches=1,
      save_image_dir=os.path.join('/tmp/eval_with_attacks', attack_name))

    preds = np.argmax(logits, axis=1)
    correct = np.equal(preds, labels).astype(np.float32)
    correct_fracs = np.sum(correct, axis=0) / len(labels)
    print("Top1 accuracy under %s: %.3f" % (attack_name, correct_fracs))


def main():
  tcu_dataset_iter = get_torch_imagenet_dataset_iter(batch_size=32)
  # model_fn = get_torch_imagenet_model()
  model_fn = get_keras_imagenet_model()
  evaluate_imagenet_model(model_fn, tcu_dataset_iter)


if __name__ == '__main__':
  main()
