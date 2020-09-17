from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
  with open(os.devnull, "w") as devnull:
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:
      yield
    finally:
      sys.stdout = old_stdout
