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


x_input = tf.placeholder(tf.float32, (None, 28, 28, 1))
y_input = tf.placeholder(tf.int64, [None])


def iter_mnist_testset(num_datapoints, batch_size, class1=7, class2=6):
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


def main(_):
  with tf.Session() as sess:
    model = mnist_convnet.Model(FLAGS.model_dir, sess)
    tcu_model = TCUWrapper(model)
    logits = tcu_model(x_input)

    def np_tcu_model(x):
      return sess.run(logits, {x_input: x})

    eval_kit.evaluate_tcu_mnist_model(
      np_tcu_model,
      iter_mnist_testset(num_datapoints=128, batch_size=128))


if __name__ == "__main__":
  tf.app.run()
