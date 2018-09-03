import numpy as np
from unrestricted_advex import eval_kit


def test_readme_example():
  def my_very_robust_model(images_batch_nchw):
    """ This function implements a valid unrestricted advex defense.

    :param images_batch_nchw: A numpy array representing a batch of images
    :return: A batch of logits
    """
    batch_size = len(images_batch_nchw)
    logits_np = np.random.randn(batch_size, 2)
    return logits_np

  eval_kit.evaluate_bird_or_bicycle_model(my_very_robust_model)


if __name__ == '__main__':
  test_readme_example()