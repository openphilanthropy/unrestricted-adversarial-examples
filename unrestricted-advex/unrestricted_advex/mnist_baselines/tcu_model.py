import tensorflow as tf


class TCUWrapper(object):
  def __init__(self, mnist_model, classes=(6, 7)):
    self.mnist_model = mnist_model
    self.classes = classes

  def __call__(self, xs):
    logits = self.mnist_model(xs)
    return tf.stack([logits[:, self.classes[0]], logits[:, self.classes[1]]], axis=1)
