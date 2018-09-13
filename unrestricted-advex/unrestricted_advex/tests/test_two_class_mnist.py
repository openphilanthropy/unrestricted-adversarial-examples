import numpy as np
from unrestricted_advex import attacks, eval_kit
from unrestricted_advex.mnist_baselines import mnist_utils


def test_two_class_mnist():
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

  # Test it on small attacks *on the training set*
  model_fn = mnist_utils.np_two_class_mnist_model(model_dir)

  # TODO: Refactor to use attack names
  attack_list = {
    'null_attack': attacks.NullAttack(),

    'random_spatial_attack': attacks.RandomSpatialAttack(
      image_shape_hwc=(28, 28, 1),
      spatial_limits=[5, 5, 5],
      black_border_size=4,
    ),

    'spatial_grid_attack': attacks.SpatialGridAttack(
      image_shape_hwc=(28, 28, 1),
      spatial_limits=[5, 5, 5],
      grid_granularity=[4, 4, 4],
      black_border_size=4,
      valid_check=mnist_utils.mnist_valid_check
    ),

    'spsa': attacks.SpsaAttack(
      model=model_fn,
      image_shape_hwc=(28, 28, 1),
      epsilon=0.3,
    ),

    'spsa_with_random_spatial': attacks.SpsaWithRandomSpatialAttack(
      model=model_fn,
      image_shape_hwc=(28, 28, 1),
      epsilon=0.3,
      spatial_limits=[5, 5, 5],
      black_border_size=4),
  }

  results = {}
  for (name, attack) in attack_list.items():
    two_class_iter = mnist_utils.two_class_iter(
      mnist.train.images, mnist.train.labels,
      num_datapoints=num_datapoints, batch_size=batch_size)
    _, _, correct, _ = eval_kit.run_attack(
      model_fn, two_class_iter, attack)
    num_correct = np.sum(correct)
    results[name] = num_correct
    print("Accuracy on %s: %s" % (name, float(num_correct) / num_datapoints))

  assert results['null_attack'] >= (num_datapoints - 1)
  assert results['spatial_grid_attack'] <= num_datapoints * 0.7
  assert results['random_spatial_attack'] <= num_datapoints * 0.9


if __name__ == '__main__':
  test_two_class_mnist()
