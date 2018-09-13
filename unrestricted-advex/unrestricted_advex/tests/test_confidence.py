import numpy as np
from numpy.testing import assert_approx_equal
from unrestricted_advex import eval_kit, attacks

batch_size = 10000
attack_list = [
  attacks.CleanData(),
]


def always_one_model_fn(x):
  logits_np = np.array([[-5.0, 5.0]] * batch_size)
  return logits_np.astype(np.float32)


def random_model_fn(x):
  logits_np = np.random.randn(batch_size, 2)
  return logits_np.astype(np.float32)


def test_confidence_always_right():
  fake_data_iter = [[
    np.zeros([batch_size, 28, 28, 1]),
    np.ones(batch_size)
  ]]

  results = eval_kit.evaluate_two_class_unambiguous_model(
    model_fn=always_one_model_fn,
    data_iter=fake_data_iter,
    model_name='always_right_model',
    attack_list=attack_list
  )

  assert results['clean']['accuracy@100'] == 1.0
  assert results['clean']['accuracy@80'] == 1.0


def test_confidence_always_wrong():
  fake_data_iter = [[
    np.zeros([batch_size, 28, 28, 1]),
    np.zeros(batch_size)
  ]]

  results = eval_kit.evaluate_two_class_unambiguous_model(
    model_fn=always_one_model_fn,
    data_iter=fake_data_iter,
    model_name='always_wrong_model',
    attack_list=attack_list
  )

  assert results['clean']['accuracy@100'] == 0.0
  assert results['clean']['accuracy@80'] == 0.0


def test_confidence_random():
  fake_data_iter = [[
    np.zeros([batch_size, 28, 28, 1]),
    np.zeros(batch_size)
  ]]

  results = eval_kit.evaluate_two_class_unambiguous_model(
    model_fn=random_model_fn,
    data_iter=fake_data_iter,
    model_name='random_model',
    attack_list=attack_list
  )
  assert_approx_equal(results['clean']['accuracy@100'], 0.5,
                      significant=2)
  assert_approx_equal(results['clean']['accuracy@80'], 0.5,
                      significant=2)


if __name__ == '__main__':
  test_confidence_random()
