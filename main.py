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
from functions import assemble_matrix_from_iterables, assemble_neumann_rhs, assemble_rhs_from_iterables, \
                      stiffness_with_diffusivity_iter, mass_with_reaction_iter, poisson_rhs_iter, mass_with_reaction_iter_2
from utils.solve import solve_with_dirichlet_data


def solve_fixed_point(mesh: Triangulation, quadrule: QuadRule, u0:np.array, tol:float, alpha:float):
  error = tol + 1
  # f = lambda x: np.ones(x.shape).T
  f = lambda x: np.array([100])
  u = u0
  i = 0
  max_iter = 10000
  errors = np.zeros(max_iter)

  while (error > tol) and (i < max_iter):
    # assemble the linear system
    Aiter = stiffness_with_diffusivity_iter(mesh, quadrule)
    Miter = mass_with_reaction_iter_2(mesh, quadrule, u, alpha)
    rhsiter = poisson_rhs_iter(mesh, quadrule, f)

    # import ipdb
    # ipdb.set_trace()

    S = assemble_matrix_from_iterables(mesh, Miter, Aiter)
    rhs = assemble_rhs_from_iterables(mesh, rhsiter)

    # solve the system
    bindices = np.unique(mesh.lines)
    u_new = solve_with_dirichlet_data(S, rhs, bindices, np.zeros_like(bindices))

    # compute error
    error = np.linalg.norm(u - u_new, ord = np.inf)
    errors[i] = error

    print(i, error)

    # update u
    u = u_new
    i += 1 
  
  mesh.tripcolor(u)
  return i, errors, u


# def solve_linear_problem(mesh, quadrule, un, alpha):
#   r""" 
#     Inspired from function solve_problem_3() from the FEM code provided as part of the class
#     Computes THE P1 FEM solution of the following problem:

#       -∆u + r(x, y) u = 100    in  Ω
#             u = 0    on ∂Ω

#     where Ω = (0, 1)^2.

#     parameters
#     ----------
#     mesh_size : `float`
#       Numeric value 0 < mesh_size < 1 tuning the mesh density.
#       Smaller value => denser mesh.

#   """

#   Aiter = stiffness_with_diffusivity_iter(mesh, quadrule)
#   Miter = mass_with_reaction_iter(mesh, quadrule, un, alpha)
#   # f = 100 * np.ones((mesh.points.shape[0], 1))
#   # rhsiter = poisson_rhs_iter(mesh, quadrule, f)



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

  n = mesh.points.shape[0]
  print("Number of vertices: ", n)
  assert n >= n_min, "The number of vertices should be >= 100"

  # define the quadrature formula
  quadrule = seven_point_gauss_6()

  # define the initial guess for u (as a column vector)
  u0 = u0_val * np.ones(n)

  # QUESTION 2: fixed point method
  i, errors, u = solve_fixed_point(mesh, quadrule, u0, tol, alpha)
  print("Number of iterations: ", i)
  print("Final error: ", errors[i-1])


if __name__ == '__main__':
  main()