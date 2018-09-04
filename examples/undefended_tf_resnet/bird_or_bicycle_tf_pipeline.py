from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from absl import app as absl_app
from absl import flags
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from examples.undefended_tf_resnet import IM_SHAPE

flags.DEFINE_string(
  name='bird_or_bicycle_data',
  help='Path to load dataset from',
  default="/root/datasets/bird-or-bicycle/v0.0.2")

FLAGS = flags.FLAGS

BIRD_CLASSES = list(range(80, 100 + 1))
BICYCLE_CLASSES = [671, 444]
BIRD_CLASSES = [cls + 1 for cls in BIRD_CLASSES]
BICYCLE_CLASSES = [cls + 1 for cls in BICYCLE_CLASSES]

def _process_input_pyfunc(file):
  img = keras_image.load_img(file, target_size=IM_SHAPE)
  x = preprocess_input(np.array(img))

  class_name = file.decode('utf-8').split('/')[-2]
  assert class_name in ['bird', 'bicycle']
  if class_name == "bird":
    label = 0
  else:
    label = 1

  return x, np.array(label).astype(np.int32)


def _process_input(filename_op):
  img, label = tf.py_func(_process_input_pyfunc, [filename_op],
                          [tf.float32, tf.int32])
  img.set_shape(IM_SHAPE)
  return img, label


def input_fn(shuffle=False):
  ds = tf.data.Dataset.list_files(
    os.path.join(FLAGS.bird_or_bicycle_data, '*/*'),
    shuffle=shuffle)
  ds = ds.map(_process_input)

  ds = ds.batch(FLAGS.batch_size, drop_remainder=False)
  return ds


def main(argv):
  del argv


if __name__ == '__main__':
  absl_app.run(main)
