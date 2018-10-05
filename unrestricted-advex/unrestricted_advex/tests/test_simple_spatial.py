import numpy as np
from unrestricted_advex.attacks import SimpleSpatialAttack

n_samples = 1000


def flatten(x):
  batch_size = len(x)
  flat_x = x.reshape([batch_size, -1])
  return flat_x


def bright_pixel_model_fn(x):
  """
  If there is a bright pixel in the image returns the first class.
  Otherwise returns the second class. Spatial attack should push the
  bright pixels off of the image.
  """
  brightness_theshold = 0.95
  flat_x = flatten(x)
  first_logit = np.max(flat_x, axis=1)
  second_logit = np.ones_like(first_logit) * brightness_theshold
  res = np.stack([first_logit, second_logit], axis=1)
  return res


def test_no_transformation():
  n_samples = 100
  x_np = np.random.rand(n_samples, 2, 2, 3)
  x_np = np.array(x_np, dtype=np.float32)
  y_np = np.ones(n_samples)

  attack = SimpleSpatialAttack(
    spatial_limits=[0, 0, 0],
    grid_granularity=[1, 1, 1],
    black_border_size=0,
  )

  x_adv = attack(bright_pixel_model_fn, x_np, y_np)
  assert np.max(x_adv - x_np) < 0.00001


def test_push_pixels_off_image():
  x_np = np.random.rand(n_samples, 2, 2, 3)
  x_np = np.array(x_np, dtype=np.float32)

  # The correct answer is that they are bright
  # So the attack must push the pixels off the edge
  y_np = np.zeros(n_samples, dtype=np.uint8)

  clean_logits = bright_pixel_model_fn(x_np)
  clean_preds = np.argmax(clean_logits, axis=1)

  clean_acc = np.mean(y_np == clean_preds)

  attack = SimpleSpatialAttack(
    spatial_limits=[0.5, 0.5, 0],
    grid_granularity=[3, 3, 1],
    black_border_size=0,
  )

  x_adv = attack(bright_pixel_model_fn, x_np, y_np)

  adv_logits = bright_pixel_model_fn(x_adv)
  adv_preds = np.argmax(adv_logits, axis=1)

  adv_acc = np.mean(y_np == adv_preds)
  print("clean_mean_brightness: %s" % np.mean(np.max(flatten(x_np), axis=1)))
  print("adv_mean_brightness: %s" % np.mean(np.max(flatten(x_adv), axis=1)))
  print("clean_acc: %s" % clean_acc)
  print("adv_acc: %s" % adv_acc)

  # The attack makes the model worse by pushing pixels off the edge
  assert abs(clean_acc - 0.5) < 0.1
  assert abs(adv_acc - 0.1) < 0.1


def test_keep_pixels_on_image():
  x_np = np.random.rand(n_samples, 2, 2, 3)
  x_np = np.array(x_np, dtype=np.float32)

  # The correct answer is that they are NOT bright
  # So the attack must NOT push the pixels off the edge
  # Accuracy should not increse by much
  y_np = np.ones(n_samples, dtype=np.uint8)

  clean_logits = bright_pixel_model_fn(x_np)
  clean_preds = np.argmax(clean_logits, axis=1)

  clean_acc = np.mean(y_np == clean_preds)

  attack = SimpleSpatialAttack(
    spatial_limits=[0.5, 0.5, 0],
    grid_granularity=[3, 3, 1],
    black_border_size=0,
  )

  x_adv = attack(bright_pixel_model_fn, x_np, y_np)

  adv_logits = bright_pixel_model_fn(x_adv)
  adv_preds = np.argmax(adv_logits, axis=1)

  adv_acc = np.mean(y_np == adv_preds)
  print("clean_mean_brightness: %s" % np.mean(np.max(flatten(x_np), axis=1)))
  print("adv_mean_brightness: %s" % np.mean(np.max(flatten(x_adv), axis=1)))
  print("clean_acc: %s" % clean_acc)
  print("adv_acc: %s" % adv_acc)

  # The attack can't make the model worse, because it can only choose to
  # NOT push pixels off the edge
  assert abs(clean_acc - 0.5) < 0.1
  assert abs(adv_acc - 0.5) < 0.1


if __name__ == '__main__':
  test_push_pixels_off_image()
  print()
  test_keep_pixels_on_image()
  print()
