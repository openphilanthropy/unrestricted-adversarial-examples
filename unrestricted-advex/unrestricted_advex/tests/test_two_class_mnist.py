from unrestricted_advex import attacks, eval_kit

from unrestricted_advex.mnist_baselines import mnist_utils


def test_two_class_mnist():
  """ Train an mnist model on a subset of mnist and then evaluate it
  on small attacks *on the training set*.
  """
  model_dir = '/tmp/two-class-mnist/test'
  batch_size = 32
  dataset_total_n_batches = 1  # use a subset of the data
  train_batches = 20

  # Train a little MNIST classifier from scratch
  print("Training mnist classifier...")
  mnist = mnist_utils.mnist_dataset(one_hot=False)
  num_datapoints = batch_size * dataset_total_n_batches

  next_batch_iter = mnist_utils.two_class_iter(
    mnist.train.images, mnist.train.labels,
    num_datapoints=num_datapoints, batch_size=batch_size,
    label_scheme='one_hot', cycle=True)

  mnist_utils.train_mnist(
    model_dir,
    lambda: next(next_batch_iter),
    train_batches, "vanilla",
    save_every=(train_batches - 1),
    print_every=10)

  model_fn = mnist_utils.np_two_class_mnist_model(model_dir)
  attack_list = [
    attacks.CleanData(),

    attacks.RandomSpatialAttack(
      image_shape_hwc=(28, 28, 1),
      spatial_limits=[5, 5, 5],
      black_border_size=4,
    ),

    attacks.SpatialGridAttack(
      image_shape_hwc=(28, 28, 1),
      spatial_limits=[5, 5, 5],
      grid_granularity=[4, 4, 4],
      black_border_size=4,
      valid_check=mnist_utils.mnist_valid_check
    ),

    attacks.SpsaAttack(
      model=model_fn,
      image_shape_hwc=(28, 28, 1),
      epsilon=0.3,
    ),

    attacks.SpsaWithRandomSpatialAttack(
      model=model_fn,
      image_shape_hwc=(28, 28, 1),
      epsilon=0.3,
      spatial_limits=[5, 5, 5],
      black_border_size=4),
  ]

  two_class_iter = mnist_utils.two_class_iter(
    mnist.train.images, mnist.train.labels,
    num_datapoints=num_datapoints, batch_size=batch_size)

  results = eval_kit.evaluate_two_class_unambiguous_model(
    model_fn,
    two_class_iter,
    attack_list)

  # Make sure that clean data has high accuracy
  assert results['clean']['accuracy@100'] >= 0.9

  # Make sure that attacks reduce accuracy
  assert results['spatial_grid']['accuracy@100'] <= 0.7
  assert results['spsa']['accuracy@100'] <= 0.6
  assert results['spsa_with_random_spatial']['accuracy@100'] <= 0.5


if __name__ == '__main__':
  test_two_class_mnist()
