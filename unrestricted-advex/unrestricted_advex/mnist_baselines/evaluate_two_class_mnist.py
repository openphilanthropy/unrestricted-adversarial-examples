import tensorflow as tf
from unrestricted_advex import eval_kit
from unrestricted_advex.mnist_baselines import mnist_utils

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "/tmp/two-class-mnist/vanilla",
                    "Where to load the model to attack from")


def main(_):
  mnist = mnist_utils.mnist_dataset(one_hot=False)
  eval_kit.evaluate_two_class_mnist_model(
    mnist_utils.np_two_class_mnist_model(FLAGS.model_dir),
    mnist_utils.two_class_iter(
      images=mnist.test.images, labels=mnist.test.labels,
      num_datapoints=128, batch_size=128))


if __name__ == "__main__":
  tf.app.run()
