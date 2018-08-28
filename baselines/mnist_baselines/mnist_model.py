import tensorflow as tf
import numpy as np

class Model(object):
  def __init__(self, restore=None, sess=None, tiny=False, name='model'):
    self.tiny = tiny
    self.name = name
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      self._build_model(tf.constant(np.zeros((1,28,28,1)), dtype=tf.float32))
    if restore:
      path = tf.train.latest_checkpoint(restore)
      saver = tf.train.Saver()
      saver.restore(sess, path)
          
  def __call__(self, xs):
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      return self._build_model(xs)

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self, x=None):
    start = 8 if self.tiny else 32

    x = tf.nn.relu(self._conv('conv1', x, 5, 1, start, self._stride_arr(1)))
    x = self._max_pool_2x2(x)
    
    x = tf.nn.relu(self._conv('conv2', x, 5, start, start*2, self._stride_arr(1)))
    x = self._max_pool_2x2(x)

    x = tf.reshape(x, [-1, 7*7*start*2])

    x = self._fully_connected(x, 64 if self.tiny else 512)

    with tf.variable_scope('logit'):
      pre_softmax = self._fully_connected(x, 10)

    return pre_softmax

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
	'DW', [filter_size, filter_size, in_filters, out_filters],
	tf.float32, initializer=tf.random_normal_initializer(
	  stddev=np.sqrt(2.0/n)))
      res = tf.nn.conv2d(x, kernel, strides, padding='SAME')
      bias = tf.get_variable('B', [out_filters],
	                     tf.float32,
                             initializer=tf.random_normal_initializer(
	                       stddev=np.sqrt(2.0/n)))
      return res+bias

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
      x = tf.reshape(x, [tf.shape(x)[0], -1])
      w = tf.get_variable(
	'DW', [prod_non_batch_dimensions, out_dim],
	initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
      b = tf.get_variable('biases', [out_dim],
			  initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _max_pool_2x2(self, x):
    return tf.nn.max_pool(x,
	                  ksize = [1,2,2,1],
			  strides=[1,2,2,1],
			  padding='SAME')
											    
