# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup

setup(
  name='bird-or-bicycle',
  version='0.0.3',
  description='Two-class unambiguous images. Follow the same dimensionality as ILSVRC 2012',
  author='Tom Brown',
  author_email='tombrown@google.com',
  url='https://github.com/google/unrestricted-adversarial-examples/bird-or-bicycle',
  packages=['bird_or_bicycle'],
  install_requires=[
    # Use semantic versioning of dependencies
    # https://stackoverflow.com/questions/39590187/in-requirements-txt-what-does-tilde-equals-mean
    'awscli ~= 1.0',
    'torchvision ~= 0.2.0',
    'Pillow ~= 5.2'
  ],
  scripts=[
    'bin/bird-or-bicycle-download',
    'bin/bird-or-bicycle-verify',
  ]
)
