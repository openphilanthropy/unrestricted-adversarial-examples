import tensorflow as tf
import numpy as np
import scipy.misc

from tensorflow.examples.tutorials.mnist import input_data

import mnist_model
import tcu_model

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

x_input = tf.placeholder(tf.float32, (None, 28, 28, 1))
y_input = tf.placeholder(tf.int64, [None])

import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from cleverhans.attacks import MadryEtAl, CarliniWagnerL2
from cleverhans.model import CallableModelWrapper

import mnist_model
import tcu_model
import time

def show(img):
  remap = " .*#"+"#"*100
  img = (img.flatten())*3
  print("START")
  for i in range(28):
    print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

images = tf.placeholder(tf.float32, (None, 28, 28, 1))
angle = tf.placeholder(tf.float32, [])
shift_xy = tf.placeholder(tf.float32, [None, 2])

padded = tf.pad(images, [(0, 0), (2, 2), (2, 2), (0, 0)])
rotated = tf.contrib.image.rotate(padded, angle, 'BILINEAR')
shifted = tf.contrib.image.translate(padded, shift_xy, 'BILINEAR')

def did_extrude(image):
  image = np.copy(image)
  image[:,2:30,2:30] = 0
  return np.any(image,axis=(1,2,3))


with tf.Session() as sess:
  mnist_model = mnist_model.Model("models/adversarial_bs=256/", sess)
  tcu_model = tcu_model.TCUWrapper(mnist_model)
  
  logits = tcu_model(x_input)
  
  which = (mnist.test.labels==7)|(mnist.test.labels==6)
  use = mnist.test.images[which].reshape((-1,28,28,1))
  use_labs = mnist.test.labels[which]==7
  
  preds = sess.run(logits, {x_input: use,
                            y_input: mnist.test.labels[which]})

  for image,image_lab in zip(use,use_labs):
    now = time.time()
    for alpha in np.arange(-3.14159/12, 3.14159/12, 3.14159/30):
      roted = sess.run(rotated, {images: [image], angle: alpha})
      ok = ~did_extrude(roted)
      if ok[0] == False:
        print("It extruded after rotation")
        continue
      roted = roted[0, 2:30, 2:30]
            
      dydx = [(dy,dx) for dx in np.arange(-10,10,.25) for dy in np.arange(-10,10,.25)]
      results = sess.run(shifted, {images: [roted]*len(dydx),
                                   shift_xy: dydx})
      ok = ~did_extrude(results)
      results = results[ok][:, 2:30, 2:30]
      lab = np.argmax(sess.run(logits, {x_input: results}),axis=1)
      
      print('lab',image_lab, 'mean',np.mean(lab))
      
      for img,l in zip(results,lab):
        if image_lab != l:
          show(img)
          break
      if 1e-8 < np.mean(lab) < 1-1e-8:
        print("Image incorrect")
        break
    else:
      print("Image was correct under all rotations and translations")
    print('delta',time.time()-now)
