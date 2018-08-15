import sys

py_version = (sys.version_info.major, sys.version_info.minor)
from setuptools import setup

setup(
  name='adv_mnist',
  packages=['adv_mnist'],
  version='0.0.1',
  install_requires=[

  ],
  author='Tom B Brown',
  author_email='tombrown@google.com',
  extras_require={
    "test": [

    ]
  },
  scripts=[

  ]
)
