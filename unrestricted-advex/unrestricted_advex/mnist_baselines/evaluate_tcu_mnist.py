import math

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from unrestricted_advex import eval_kit
from unrestricted_advex.mnist_baselines import mnist_convnet
from unrestricted_advex.mnist_baselines.tcu_model import TCUWrapper

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "/tmp/tcu-mnist/vanilla",
                    "Where to put the trained model checkpoint")

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

x_input = tf.placeholder(tf.float32, (None, 28, 28, 1))
y_input = tf.placeholder(tf.int64, [None])


def show(img):
  remap = " .*#" + "#" * 100
  img = (img.flatten()) * 3
  print("START")
  for i in range(28):
    print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))


def main(_):
  with tf.Session() as sess:
    model = mnist_convnet.Model(FLAGS.model_dir, sess)
    # mnist_model = mnist_model.Model("models/clean/", sess)
    tcu_model = TCUWrapper(model)

    def iterator(num_datapoints, batch_size):
      which = (mnist.test.labels == 7) | (mnist.test.labels == 6)
      num_batches = math.ceil(num_datapoints / batch_size)
      for i in range(int(num_batches)):
        images = mnist.test.images[which][i:i + batch_size].reshape((batch_size, 28, 28, 1))
        labels = mnist.test.labels[which][i:i + batch_size] == 7
        yield images, labels

    logits = tcu_model(x_input)

    def np_tcu_model(x):
      return sess.run(logits, {x_input: x})

    eval_kit.evaluate_tcu_mnist_model(
      np_tcu_model,
      iterator(num_datapoints=128, batch_size=128))


if __name__ == "__main__":
  tf.app.run()
