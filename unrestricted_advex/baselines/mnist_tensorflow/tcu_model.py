import tensorflow as tf
import numpy as np

class TCUWrapper(object):
  def __init__(self, mnist_model, classes=(6,7)):
    self.mnist_model = mnist_model
    self.classes = classes

  def __call__(self, xs):
    logits = self.mnist_model(xs)
    class_a = logits[:,self.classes[0]]
    class_b = logits[:,self.classes[1]]
    return class_a - class_b
