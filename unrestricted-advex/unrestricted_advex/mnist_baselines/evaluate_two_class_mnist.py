import math

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from unrestricted_advex import eval_kit
from unrestricted_advex.mnist_baselines import mnist_convnet

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "/tmp/two-class-mnist/vanilla",
                    "Where to put the trained model checkpoint")


class TwoClassWrapper(object):
  def __init__(self, mnist_model, classes=(6, 7)):
    self.mnist_model = mnist_model
    self.classes = classes

  def __call__(self, xs):
    logits = self.mnist_model(xs)
    return tf.stack([logits[:, self.classes[0]], logits[:, self.classes[1]]], axis=1)


def two_class_mnist_iter(num_datapoints, batch_size, class1=7, class2=6):
  """Filter MNIST to only two classes (e.g. sixes and sevens)"""
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
  which = (mnist.test.labels == class1) | (mnist.test.labels == class2)
  images_2class = mnist.test.images[which].astype(np.float32)
  labels_2class = mnist.test.labels[which]
  num_batches = math.ceil(num_datapoints / batch_size)
  for i in range(int(num_batches)):
    images = images_2class[i:i + batch_size].reshape((batch_size, 28, 28, 1))
    labels = labels_2class[i:i + batch_size] == class1
    yield images, labels


def show(img):
  remap = " .*#" + "#" * 100
  img = (img.flatten()) * 3
  print("START")
  for i in range(28):
    print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))


def np_two_class_mnist_model(model_dir):
  with tf.Graph().as_default():
    x_input = tf.placeholder(tf.float32, (None, 28, 28, 1))
    sess = tf.Session()
    model = mnist_convnet.Model(model_dir, sess)
    two_class_model = TwoClassWrapper(model)
    logits = two_class_model(x_input)

    def np_model(x):
      return sess.run(logits, {x_input: x})

    return np_model

def main(_):
  eval_kit.evaluate_two_class_mnist_model(
    np_two_class_mnist_model(FLAGS.model_dir),
    two_class_mnist_iter(num_datapoints=128, batch_size=128))


if __name__ == "__main__":
  tf.app.run()
