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
        'tqdm',
        'cleverhans',
        'foolbox',
    ],
)
