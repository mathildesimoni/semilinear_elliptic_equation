import numpy as np
from scipy import sparse
from numbers import Number
from typing import Callable, Tuple

from functools import partial

# import from other python files
from functions import gauss_quadrature

gauss6 = partial(gauss_quadrature, order=6) # gauss quadrature scheme of order 6

def main():
  print("Hello!")

  # define parameters
  alpha = 0.1 # alpha = 2.0
  tol = 1e-6

  # tryout = gauss6(0, 1)
  # print(tryout)


if __name__ == '__main__':
  main()