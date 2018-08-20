import numpy as np
import tensorflow as tf
from cleverhans.attacks import SPSA
from cleverhans.model import Model
from six.moves import xrange


class CleverhansModelWrapper(Model):
  def __init__(self, model_fn):
    """
    Wrap a callable function that takes a numpy array of shape (N, C, H, W),
    and outputs a numpy vector of length N, with each element in range [0, 1].
    """
    self.nb_classes = 2
    self.model_fn = model_fn

  def fprop(self, x, **kwargs):
    logits_op = tf.py_func(self.model_fn, [x], tf.float32)
    return {'logits': logits_op}


def spsa_attack(model, batch_nchw, labels, epsilon=(16. / 255)):  # (4. / 255)):
  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=(1,) + batch_nchw.shape[1:])
    y_label = tf.placeholder(tf.int32, shape=(1,))

    cleverhans_model = CleverhansModelWrapper(model)
    attack = SPSA(cleverhans_model)

    x_adv = attack.generate(
      x_input,
      y=y_label,
      epsilon=epsilon,
      num_steps=200,
      early_stop_loss_threshold=-1.,
      spsa_samples=32,
      is_debug=True)

    # Run computation
    all_x_adv_np = []
    with tf.Session() as sess:
      for i in xrange(len(batch_nchw)):
        x_adv_np = sess.run(x_adv, feed_dict={
          x_input: np.expand_dims(batch_nchw[i], axis=0),
          y_label: np.expand_dims(labels[i], axis=0),
        })
        all_x_adv_np.append(x_adv_np)
    return np.concatenate(all_x_adv_np)


def null_attack(model, x_np, y_np):
  del model, y_np  # unused
  return x_np
