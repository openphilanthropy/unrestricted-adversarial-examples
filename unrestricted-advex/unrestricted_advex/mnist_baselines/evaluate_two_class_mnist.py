import math

import numpy as np
import tensorflow as tf
from unrestricted_advex import eval_kit
from unrestricted_advex.mnist_baselines import mnist_convnet
from unrestricted_advex.two_class_mnist_dataset import two_class_mnist_iter

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "/tmp/two-class-mnist/vanilla",
                    "Where to put the trained model checkpoint")

x_input = tf.placeholder(tf.float32, (None, 28, 28, 1))
y_input = tf.placeholder(tf.int64, [None])


class TwoClassWrapper(object):
  def __init__(self, mnist_model, classes=(6, 7)):
    self.mnist_model = mnist_model
    self.classes = classes

  def __call__(self, xs):
    logits = self.mnist_model(xs)
    return tf.stack([logits[:, self.classes[0]], logits[:, self.classes[1]]], axis=1)


def show(img):
  remap = " .*#" + "#" * 100
  img = (img.flatten()) * 3
  print("START")
  for i in range(28):
    print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))


def main(_):
  with tf.Session() as sess:
    model = mnist_convnet.Model(FLAGS.model_dir, sess)
    two_class_model = TwoClassWrapper(model)
    logits = two_class_model(x_input)

    def np_model(x):
      return sess.run(logits, {x_input: x})

    eval_kit.evaluate_two_class_mnist_model(
      np_model,
      two_class_mnist_iter(num_datapoints=4, batch_size=2))


if __name__ == "__main__":
  tf.app.run()
