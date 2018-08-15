import tensorflow as tf
import numpy as np
import scipy.misc

from tensorflow.examples.tutorials.mnist import input_data

import mnist_model
import tcu_model

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

x_input = tf.placeholder(tf.float32, (None, 28, 28, 1))
y_input = tf.placeholder(tf.int64, [None])

with tf.Session() as sess:
    mnist_model = mnist_model.Model("models/baseline", sess)
    tcu_model = tcu_model.TCUWrapper(mnist_model)
    
    logits = tcu_model(x_input)

    which = (mnist.test.labels==7)|(mnist.test.labels==6)
    
    preds = sess.run(logits, {x_input: mnist.test.images[which].reshape((-1,28,28,1)),
                              y_input: mnist.test.labels[which]})

    is_6 = (mnist.test.labels[which]==6)
    
    print(np.mean((preds>0)==is_6))

    order_6 = np.argsort(preds[is_6])
    order_7 = np.argsort(-preds[~is_6])

    scipy.misc.imsave("/tmp/6s.png", np.concatenate(mnist.test.images[which][is_6][order_6][:20].reshape((-1,28,28)),axis=1))
    scipy.misc.imsave("/tmp/7s.png", np.concatenate(mnist.test.images[which][~is_6][order_7][:20].reshape((-1,28,28)),axis=1))
