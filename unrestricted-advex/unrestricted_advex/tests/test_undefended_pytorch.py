"""Test examples/undefended_pytorch.

We only run the script for training / evaluation
for one tiny batch to verify that the program can
successfully run without issue. The correctness of
the results are not checked in this auto-test.
"""

import os


def get_example_main_path():
  return os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..',
                 'examples', 'undefended_pytorch_resnet', 'main.py'))


def test_training():
  main_py = get_example_main_path()
  assert os.system('python "{main_py}" --smoke-test'.format(
    main_py=main_py)) == 0


def test_eval():
  main_py = get_example_main_path()
  env = 'CUDA_VISIBLE_DEVICES=0'
  cmd = '{env} python "{main_py}" --evaluate --smoke-test'.format(
    env=env, main_py=main_py)
  assert os.system(cmd) == 0


if __name__ == '__main__':
  test_training()
  test_eval()
