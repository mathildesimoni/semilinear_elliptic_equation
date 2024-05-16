import numpy as np
# from scipy import sparse
from numbers import Number
from typing import Callable, Tuple
from functools import partial


def gauss_quadrature(a: Number, b: Number, order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
  """ Given the element boundaries `(a, b)`, return the weights and evaluation points
      corresponding to a gaussian quadrature scheme of order `order`.

      Parameters
      ----------

      a : `float`
        the left boundary of the element
      b : `float`
        the right boundary of the element
      order : `int`
        the order of the Gaussian quadrature scheme

      Returns
      -------

      weights : `np.ndarray`
        the weights of the quadrature scheme
      points : `np.ndarray`
        the points (abscissae) over (a, b)
  """
  assert b > a
  points, weights = np.polynomial.legendre.leggauss(order)
  points = (points + 1) / 2 # adapt to interval [0,1]
  return (b - a) / 2 * weights, a + points * (b - a) # adapt to interval [a, b]


