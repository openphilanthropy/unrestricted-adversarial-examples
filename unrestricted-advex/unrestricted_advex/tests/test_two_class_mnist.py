import pytest
import tensorflow as tf
from unrestricted_advex import eval_kit, attacks
from unrestricted_advex.mnist_baselines import mnist_utils

model_dir = '/tmp/two-class-mnist/test'


def train_overfit_classifier(num_batches, batch_size):
  print("Training mnist classifier...")

  next_batch_iter = mnist_utils.get_two_class_iterator(
    'train',
    num_datapoints=num_batches * batch_size,
    batch_size=batch_size,
    label_scheme='one_hot',
    cycle=True)

  mnist_utils.train_mnist(
    model_dir,
    lambda: next(next_batch_iter),
    num_batches, "vanilla",
    save_every=(num_batches - 1),
    print_every=10)

  return mnist_utils.np_two_class_mnist_model(model_dir)


@pytest.mark.skipif(not tf.test.is_gpu_available(),
                    reason="Running attacks on MNIST currently requires a GPU :( ")
def test_two_class_mnist_accuracy():
  """ Train an mnist model on a subset of mnist and then evaluate it
  on small attacks *on the training set*.
  """
  model_fn = train_overfit_classifier(num_batches=32, batch_size=1)
  dataset_iter = mnist_utils.get_two_class_iterator('train', num_datapoints=32, batch_size=1)

  mnist_spatial_limits = [10, 10, 10]
  mnist_shape = (28, 28, 1)
  mnist_black_border_size = 4

  attack_list = [
    attacks.CleanData(),

    attacks.SpsaAttack(
      model_fn,
      epsilon=0.3,
      image_shape_hwc=mnist_shape,
    ),

    attacks.SpsaWithRandomSpatialAttack(
      model_fn,
      epsilon=0.3,
      spatial_limits=mnist_spatial_limits,
      black_border_size=mnist_black_border_size,
      image_shape_hwc=mnist_shape,
    ),

    attacks.SimpleSpatialAttack(
      grid_granularity=[5, 5, 11],
      spatial_limits=mnist_spatial_limits,
      black_border_size=mnist_black_border_size,
    ),
  ]

  boundary_attack = attacks.BoundaryWithRandomSpatialAttack(
    model_fn,
    max_l2_distortion=4,
    label_to_examples=eval_kit._get_mnist_labels_to_examples(),
    spatial_limits=mnist_spatial_limits,
    black_border_size=mnist_black_border_size,
    image_shape_hwc=mnist_shape,
  )

  # We limit the boundary attack to the first datapoint to speed up eval
  boundary_attack._stop_after_n_datapoints = 1
  attack_list.append(boundary_attack)

  results = eval_kit.evaluate_two_class_unambiguous_model(
    model_fn,
    data_iter=dataset_iter,
    model_name="overfit_mnist",
    attack_list=attack_list)

  # Make sure that clean data has high accuracy
  assert results['clean']['accuracy@100'] >= 0.9

  # Make sure that attacks reduce accuracy
  assert results['spatial_grid']['accuracy@100'] <= 0.7
  assert results['spsa']['accuracy@100'] <= 0.6
  assert results['spsa_with_random_spatial']['accuracy@100'] <= 0.5

  # Run a smoke test on all attacks with a batch size of one
  # TODO: Split this into a separate test
  dataset_iter = mnist_utils.get_two_class_iterator('train', num_datapoints=1, batch_size=1)
  eval_kit.evaluate_two_class_mnist_model(
    model_fn,
    dataset_iter=dataset_iter,
    model_name="overfit_mnist")


if __name__ == '__main__':
  test_two_class_mnist_accuracy()
