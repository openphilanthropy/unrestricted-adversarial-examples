"""Load datasets and models for TCU-ImageNet and TCU-MNIST"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import torch
import torchvision

import tcu_images
from tcu_images import CLASS_NAME_TO_IMAGENET_CLASS, BICYCLE_IDX, BIRD_IDX


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
