"""Evaluate a model with attacks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tcu_images
import tensorflow as tf
import torch
import torchvision
from tcu_images import CLASS_NAME_TO_IMAGENET_CLASS
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
    assert x_np.shape[-1] == 3 or x_np.shape[-1] == 1, "Data was {}, should be NHWC".format(x_np.shape)
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

  assert train_dataset.class_to_idx['bicycle'] == 0
  assert train_dataset.class_to_idx['bird'] == 1

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

  k_model = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000)

  def model_wrapper(x_np):
    # it seems keras pre-trained model directly output softmax-ed probs
    x_np = preprocess_input(x_np * 255)

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


def evaluate_tcu_model(model_fn, dataset_iter, attack_list):
  for (attack_fn, attack_name) in attack_list:
    print("Executing attack: %s" % attack_name)
    logits, labels = run_attack(
      model_fn, dataset_iter, attack_fn, max_num_batches=1,
      save_image_dir=os.path.join('/tmp/eval_with_attacks', attack_name))
    preds = (logits[:, 0] < logits[:, 1]).astype(np.int64)
    correct = np.equal(preds, labels).astype(np.float32)
    correct_fracs = np.sum(correct, axis=0) / len(labels)
    print("Fraction correct under %s: %.3f" % (attack_name, correct_fracs))

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


def evaluate_images_tcu_model(model_fn, dataset_iter):
  return evaluate_tcu_model(model_fn, dataset_iter, [
    # (attacks.null_attack, 'null_attack'),
    (attacks.spatial_attack, 'spatial_attack'),
    # (attacks.spsa_attack, 'spsa_attack'),
  ])
    
    
def main():
  tcu_dataset_iter = get_torch_tcu_dataset_iter(batch_size=64, shuffle=True)
  model_fn = get_keras_tcu_model()
  evaluate_images_tcu_model(model_fn, tcu_dataset_iter)


if __name__ == '__main__':
  main()
