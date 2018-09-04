import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data


CLASS1 = 6
CLASS2 = 7

def two_class_mnist_dataset():
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
  which = (mnist.test.labels == CLASS1) | (mnist.test.labels == CLASS2)
  images_2class = mnist.test.images[which].astype(np.float32)
  labels_2class = mnist.test.labels[which]
  return images_2class.reshape((-1, 28, 28, 1)), labels_2class == CLASS2

def two_class_mnist_iter(num_datapoints, batch_size):
  """Filter MNIST to only sevens and eights"""
  images_2class, labels_2class = two_class_mnist_dataset()
  num_batches = math.ceil(num_datapoints / batch_size)
  for i in range(int(num_batches)):
    images = images_2class[i:i + batch_size]
    labels = labels_2class[i:i + batch_size]
    yield images, labels

    
