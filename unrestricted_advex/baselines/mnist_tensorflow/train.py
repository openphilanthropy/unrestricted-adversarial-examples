import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from cleverhans.attacks import MadryEtAl

from mnist_model import Model

def show(img):
    remap = " .*#"+"#"*100
    img = (img.flatten())*3
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))
                                

batch_size = 128
train_mode = "adversarial" # 
model_dir = "models/adversarial"

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
train_step = tf.train.AdamOptimizer(5e-3).minimize(loss,
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
      #x_batch_adv = np.clip(x_batch + np.random.uniform(-.7, .7, size=x_batch.shape), 0, 1)
      #print(np.sum((x_batch-x_batch_adv)**2,axis=(1,2,3)))

      #show(x_batch[0])
      #show(x_batch_adv[0])

      """
      nat_dict = {x_input: x_batch,
                  y_input: y_batch}
      
      adv_dict = {x_input: x_batch_adv,
                  y_input: y_batch}

      a,l,_ = sess.run((accuracy, loss, nat_summaries), nat_dict)
      print(batch_num,"Clean accuracy", a, "loss", l)
      a,l,_ = sess.run((accuracy, loss, adv_summaries), adv_dict)
      print(batch_num,"Adv accuracy", a, "loss", l)
      exit(0)
      """
      
    else:
      x_batch_adv = x_batch
        
    nat_dict = {x_input: x_batch,
                y_input: y_batch}

    adv_dict = {x_input: x_batch_adv,
                y_input: y_batch}

    if batch_num%100 == 0:
      a,l,_ = sess.run((accuracy, loss, nat_summaries), nat_dict)
      print(batch_num,"Clean accuracy", a, "loss", l)
      if train_mode == "adversarial":
        a,l,_ = sess.run((accuracy, loss, adv_summaries), adv_dict)
        print(batch_num,"Adv accuracy", a, "loss", l)
      

    if batch_num%1000 == 0:
        saver.save(sess, os.path.join(model_dir, "checkpoint"),
                   global_step=global_step)

    sess.run(train_step, nat_dict)
    sess.run(train_step, adv_dict)
