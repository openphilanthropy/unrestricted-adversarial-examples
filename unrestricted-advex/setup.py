from setuptools import setup

setup(
  name='unrestricted_advex',
  version='0.0.3',
  description='Evaluation code and mnist_baselines for the warmup to the unrestricted adversarial examples contest',
  author='Tom Brown',
  author_email='tombrown@google.com',
  url='https://github.com/google/unrestricted-advex',
  packages=['unrestricted_advex'],
  install_requires=[
    # Use semantic versioning of dependencies
    # https://stackoverflow.com/questions/39590187/in-requirements-txt-what-does-tilde-equals-mean
    'tqdm ~= 4.0',
    'cleverhans ~= 2.1',
    'foolbox ~= 1.3',
    'randomgen',
    'terminaltables ~= 3.1',
  ],
  # Explicit dependence on TensorFlow is not supported.
  # See https://github.com/tensorflow/tensorflow/issues/7166
  extras_require={
    'tf': ['tensorflow>=1.0.0'],
    'tf_gpu': ['tensorflow-gpu>=1.0.0'],
    'test': [
      'pytest',
      'keras',
    ],
    'pytorch': ['torch==0.4.0', 'torchvision==0.2.1'],
  },
)
