import mnist_model
import tcu_model
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from unrestricted_advex import eval_kit

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

x_input = tf.placeholder(tf.float32, (None, 28, 28, 1))
y_input = tf.placeholder(tf.int64, [None])


def show(img):
  remap = " .*#"+"#"*100
  img = (img.flatten())*3
  print("START")
  for i in range(28):
    print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


with tf.Session() as sess:
  mnist_model = mnist_model.Model("models/adversarial_bs=256/", sess)
  #mnist_model = mnist_model.Model("models/clean/", sess)
  tcu_model = tcu_model.TCUWrapper(mnist_model)

  def iterator():
    which = (mnist.test.labels==7)|(mnist.test.labels==6)
    for i in range(1):
      yield mnist.test.images[which][i:i+100].reshape((100,28,28,1)), mnist.test.labels[which][i:i+100]==7


  logits = tcu_model(x_input)
  def np_tcu_model(x):
    return sess.run(logits, {x_input: x})

  eval_kit.evaluate_tcu_mnist_model(np_tcu_model,
                                    iterator())
