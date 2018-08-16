import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from cleverhans.attacks import MadryEtAl

from mnist_model import Model

batch_size = 256
train_mode = "vanilla"  # "adversarial" #
model_dir = "models/"+train_mode+"_bs=256"

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x_input = tf.placeholder(tf.float32, (None, 28, 28, 1))
y_input = tf.placeholder(tf.float32, [None, 10])

model = Model()
logits = model(x_input)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_input,
                                                      logits=logits)
loss = tf.reduce_mean(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1),
                                           tf.argmax(y_input, axis=1)),
                                  dtype=tf.float32))

global_step = tf.contrib.framework.get_or_create_global_step()
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss,
                                                   global_step=global_step)


saver = tf.train.Saver(max_to_keep=3)
a=tf.summary.scalar('accuracy adv train', accuracy)
b=tf.summary.scalar('xent adv train', loss)
c=tf.summary.image('images adv train', x_input)
adv_summaries = tf.summary.merge([a,b,c])

a=tf.summary.scalar('accuracy nat train', accuracy)
b=tf.summary.scalar('xent nat train', loss)
c=tf.summary.image('images nat train', x_input)
nat_summaries = tf.summary.merge([a,b,c])

with tf.Session() as sess:
  attack = MadryEtAl(model, sess=sess)

  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  for batch_num in range(1000000):
    x_batch, y_batch = mnist.train.next_batch(batch_size)
    x_batch = np.reshape(x_batch, (-1, 28, 28, 1))

    if train_mode == "adversarial" and batch_num>1000:
      x_batch_adv = attack.generate_np(x_batch, y=y_batch, eps=.3,
                                       nb_iter=40, eps_iter=.01,
                                       rand_init=True,
                                       clip_min=0, clip_max=1)

    else:
      x_batch_adv = x_batch

    nat_dict = {x_input: x_batch,
                y_input: y_batch}

    adv_dict = {x_input: x_batch_adv,
                y_input: y_batch}

    if batch_num%100 == 0:
      a,l,s = sess.run((accuracy, loss, nat_summaries), nat_dict)
      summary_writer.add_summary(s, sess.run(global_step))
      print(batch_num,"Clean accuracy", a, "loss", l)
      if train_mode == "adversarial":
        a,l,s = sess.run((accuracy, loss, adv_summaries), adv_dict)
        summary_writer.add_summary(s, sess.run(global_step))
        print(batch_num,"Adv accuracy", a, "loss", l)


    if batch_num%1000 == 0:
        saver.save(sess, os.path.join(model_dir, "checkpoint"),
                   global_step=global_step)

    sess.run(train_step, nat_dict)
    sess.run(train_step, adv_dict)
