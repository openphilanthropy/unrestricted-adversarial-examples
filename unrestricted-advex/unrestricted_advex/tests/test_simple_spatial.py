import numpy as np


def bright_pixel_model_fn(x):
  """
  If there is a bright pixel in the image returns the first class.
  Otherwise returns the second class. Spatial attack should push the
  bright pixels off of the image.
  """
  flat_x = np.ravel(x)
  first_logit = np.max(flat_x, axis=1)
  second_logit = np.ones_like(first_logit) * 0.5
  res = np.stack([second_logit, first_logit], axis=1)
  return res


def test_no_transformation(self):
  x_val = np.random.rand(100, 2, 2, 1)
  x_val = np.array(x_val, dtype=np.float32)

  x_adv_p = self.attack.generate(x, batch_size=100, dx_min=0.0,
                                 dx_max=0.0, n_dxs=1, dy_min=0.0,
                                 dy_max=0.0, n_dys=1, angle_min=0,
                                 angle_max=0, n_angles=1)

  x_adv = self.sess.run(x_adv_p, {x: x_val})
  self.assertClose(x_adv, x_val)


def test_push_pixels_off_image(self):
  x_val = np.random.rand(100, 2, 2, 1)
  x_val = np.array(x_val, dtype=np.float32)

  # The correct answer is that they are bright
  # So the attack must push the pixels off the edge
  y = np.zeros([100, 2])
  y[:, 0] = 1.

  x = tf.placeholder(tf.float32, shape=(None, 2, 2, 1))
  x_adv_p = self.attack.generate(x,
                                 y=y, batch_size=100, dx_min=-0.5,
                                 dx_max=0.5, n_dxs=3, dy_min=-0.5,
                                 dy_max=0.5, n_dys=3, angle_min=0,
                                 angle_max=0, n_angles=1)
  x_adv = self.sess.run(x_adv_p, {x: x_val})

  old_labs = np.argmax(y, axis=1)
  new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)
  print(np.mean(old_labs == new_labs))
  self.assertTrue(np.mean(old_labs == new_labs) < 0.3)


def test_keep_pixels_on_image(self):
  x_val = np.random.rand(100, 2, 2, 1)
  x_val = np.array(x_val, dtype=np.float32)

  # The correct answer is that they are NOT bright
  # So the attack must NOT push the pixels off the edge
  y = np.zeros([100, 2])
  y[:, 0] = 1.

  x = tf.placeholder(tf.float32, shape=(None, 2, 2, 1))
  x_adv_p = self.attack.generate(x,
                                 y=y, batch_size=100, dx_min=-0.5,
                                 dx_max=0.5, n_dxs=3, dy_min=-0.5,
                                 dy_max=0.5, n_dys=3, angle_min=0,
                                 angle_max=0, n_angles=1)
  x_adv = self.sess.run(x_adv_p, {x: x_val})

  old_labs = np.argmax(y, axis=1)
  new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)
  print(np.mean(old_labs == new_labs))
  self.assertTrue(np.mean(old_labs == new_labs) < 0.3)


if __name__ == '__main__':
  test_two_class_mnist_accuracy()
