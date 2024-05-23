import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from numbers import Number
from typing import Callable, Tuple

from functools import partial

# import from other python files
from utils.quad import seven_point_gauss_6
from utils.mesh import Triangulation
from utils.quad import QuadRule
from functions import *


def solve_fixed_point(mesh: Triangulation, quadruple: QuadRule, u0:np.array, tol:float, alpha:float):
  return 0

def solve_linear_problem(mesh, quadrule, un, alpha):
  r""" 
    Inspired from function solve_problem_3() from the FEM code provided as part of the class
    Computes THE P1 FEM solution of the following problem:

      -∆u + r(x, y) u = 100    in  Ω
            u = 0    on ∂Ω

    where Ω = (0, 1)^2.

    parameters
    ----------
    mesh_size : `float`
      Numeric value 0 < mesh_size < 1 tuning the mesh density.
      Smaller value => denser mesh.

  """

  Aiter = stiffness_with_diffusivity_iter(mesh, quadrule)
  Miter = mass_with_reaction_iter(mesh, quadrule, un, alpha)
  # f = 100 * np.ones((mesh.points.shape[0], 1))
  # rhsiter = poisson_rhs_iter(mesh, quadrule, f)



# def solve_problem_3(mesh_size=0.01):
  
#   from quad import seven_point_gauss_6
#   from scipy.sparse import linalg as splinalg

#   # make the square domain with mesh size `mesh_size`
#   square = np.array([ [0, 0],
#                       [1, 0],
#                       [1, 1],
#                       [0, 1] ])
#   mesh = Triangulation.from_polygon(square, mesh_size=mesh_size)

#   quadrule = seven_point_gauss_6()

#   # define the diffusivity and reactivity as a function of x with
#   # shape (nquadpoints, 2)
#   fdiffuse = lambda x: 1 + x[:, 0]
#   freact = lambda x: 4 * np.pi**2 * (1 + x[:, 0])

#   def f(x: np.ndarray) -> np.ndarray:
#     x, y = x.T
#     return 2 * np.pi * np.cos(2 * np.pi * y) * \
#            (np.sin(2 * np.pi * x) + 6 * np.pi * (1 + x) * np.cos(2 * np.pi * x))

#   Aiter = stiffness_with_diffusivity_iter(mesh, quadrule, fdiffuse=fdiffuse)
#   Miter = mass_with_reaction_iter(mesh, quadrule, freact=freact)
#   rhsiter = poisson_rhs_iter(mesh, quadrule, f)

#   S = assemble_matrix_from_iterables(mesh, Miter, Aiter)
#   rhs = assemble_rhs_from_iterables(mesh, rhsiter)

#   solution_approx = splinalg.spsolve(S, rhs)

#   mesh.tripcolor(solution_approx)

#   exact = lambda x: np.cos(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])
#   dexact = lambda x: np.stack([ -2 * np.pi * np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1]),
#                                 -2 * np.pi * np.cos(2 * np.pi * x[:, 0]) * np.sin(2 * np.pi * x[:, 1]) ], axis=1)

#   dnorm = compute_H1_norm_difference(mesh, quadrule, solution_approx, exact, dexact)

#   print('The H1-norm of the difference between the approximate and the exact solution equals {:.6}.'.format(dnorm))






def main():

  # define parameters
  alpha = 0.1 # OR alpha = 2.0
  tol = 1e-6 # tolerance for the fixed point method
  u0_val = 0 # initial solution
  n_min = 100  # minimum number of vertices

  # build the mesh
  mesh_size = 0.1 # 0.2 gives 45 < 100 vertices, 0.1 gives 142 > 100 vertices 
  square = np.array([ [0, 0],
                      [1, 0],
                      [1, 1],
                      [0, 1] ]) 
  mesh = Triangulation.from_polygon(square, mesh_size=mesh_size) # make the square domain with mesh size `mesh_size`
  mesh.plot()
  print(mesh.triangles)

  n = mesh.points.shape[0]
  print("Number of vertices: ", n)
  assert n >= n_min, "The number of vertices should be >= 100"

  # define the quadrature formula
  quadrule = seven_point_gauss_6()

  # define the initial guess for u (as a column vector)
  u0 = u0_val * np.ones((n, 1))

  # QUESTION 2: fixed point method
  # solve_fixed_point(mesh, quadrule, u0, tol, alpha)






if __name__ == '__main__':
  main()