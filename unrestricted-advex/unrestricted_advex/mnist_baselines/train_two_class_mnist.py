import tensorflow as tf
from unrestricted_advex.mnist_baselines import mnist_utils

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_mode", "vanilla", "Model type (either 'vanilla' or 'adversarial')")
flags.DEFINE_string("model_dir", "/tmp/two-class-mnist/vanilla",
                    "Where to put the trained model checkpoint")
flags.DEFINE_integer("batch_size", 128, "Batch size for training the model")
flags.DEFINE_integer("total_batches", 1000000, "Total number of batches to train for")


def main(_):
  assert FLAGS.train_mode in ['vanilla', 'adversarial']
  mnist = mnist_utils.mnist_dataset(one_hot=True)
  next_batch_fn = lambda: mnist.train.next_batch(FLAGS.batch_size)
  mnist_utils.train_mnist(FLAGS.model_dir, next_batch_fn,
                          FLAGS.total_batches, FLAGS.train_mode)


if __name__ == "__main__":
  tf.app.run()
