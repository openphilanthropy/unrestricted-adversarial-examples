import tensorflow as tf
from unrestricted_advex import eval_kit
from unrestricted_advex.mnist_baselines import mnist_utils

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "/tmp/two-class-mnist/vanilla",
                    "Where to load the model to attack from")
flags.DEFINE_integer("num_datapoints", 128,
                     "How many datapoints to evaluate on")
flags.DEFINE_integer("batch_size", 128,
                     "Batch size to use during evaluation")


def main(_):
  dataset_iter = mnist_utils.get_two_class_iterator(
    'test',
    num_datapoints=FLAGS.num_datapoints,
    batch_size=FLAGS.batch_size)

  eval_kit.evaluate_two_class_mnist_model(
    model_fn=mnist_utils.np_two_class_mnist_model(FLAGS.model_dir),
    dataset_iter=dataset_iter,
    model_name='example_mnist_model')

if __name__ == "__main__":
  tf.app.run()
