from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from bird_or_bicyle import CLASS_NAME_TO_IMAGENET_CLASS
from tensorflow.keras.applications.resnet50 import preprocess_input
from unrestricted_advex import eval_kit

if __name__ == '__main__':
  tf.keras.backend.set_image_data_format('channels_last')

  # Variable scoping lets us use our _graph in our model_fn
  _graph = tf.Graph()
  with _graph.as_default():
    k_model = tf.keras.applications.resnet50.ResNet50(
      include_top=True, weights='imagenet', input_tensor=None,
      input_shape=None, pooling=None, classes=1000)


  def model_fn(x_np):
    """A normal keras resnet that was pretrained on ImageNet"""
    x_np = preprocess_input(x_np * 255)

    with _graph.as_default():
      prob1000 = k_model.predict_on_batch(x_np) / 10

    # Keras returns softmax-ed probs, we convert them back to logits
    fake_logits1000 = np.log(prob1000)

    bird_max_logit = np.max(
      fake_logits1000[:, CLASS_NAME_TO_IMAGENET_CLASS['bird']], axis=1)
    bicycle_max_logit = np.max(
      fake_logits1000[:, CLASS_NAME_TO_IMAGENET_CLASS['bicycle']], axis=1)

    two_class_logits = np.concatenate((
      bicycle_max_logit[:, None],
      bird_max_logit[:, None]),
      axis=1)
    return two_class_logits


  eval_kit.evaluate_bird_or_bicycle_model(model_fn)
