import itertools

import bird_or_bicyle
import numpy as np
from unrestricted_advex import eval_kit


def get_tiny_iterator():
  """An iterator with a single sample"""
  dataset_iter = bird_or_bicyle.get_iterator('test', batch_size=1)
  return itertools.islice(dataset_iter, 1)


def test_readme_example_for_smoke():

  def my_very_robust_model(images_batch_nhwc):
    """ This function implements a valid unrestricted advex defense that always returns higher
    logits for the second class """
    batch_size = len(images_batch_nhwc)
    logits_np = np.array([[-5.0, 5.0]] * batch_size)
    return logits_np.astype(np.float32)

  eval_kit.evaluate_bird_or_bicycle_model(
    my_very_robust_model,
    dataset_iter=get_tiny_iterator())


if __name__ == '__main__':
  test_readme_example_for_smoke()
