import numpy as np
from scipy import sparse
from scipy import optimize
from scipy.sparse import linalg as splinalg
from numbers import Number
from typing import Callable, Tuple


from functools import partial

# import from other python files
from utils.quad import seven_point_gauss_6
from utils.mesh import Triangulation
from utils.quad import QuadRule
from functions import assemble_matrix_from_iterables, assemble_neumann_rhs, assemble_rhs_from_iterables, \
                      stiffness_with_diffusivity_iter, mass_with_reaction_iter, poisson_rhs_iter, mass_with_reaction_iter_2, \
                      rhs_newton, rhs_newton_a, rhs_newton_b, mass_with_reaction_iter_3
from utils.solve import solve_with_dirichlet_data

# QUESTION 2
def solve_fixed_point(mesh: Triangulation, quadrule: QuadRule, u0:np.array, tol:float, alpha:float):
  error = tol + 1
  f = lambda x: np.array([100])
  u = u0
  i = 0
  max_iter = 10000
  errors = np.zeros(max_iter)

  # assemble the RHS of the linear system only once
  rhsiter = poisson_rhs_iter(mesh, quadrule, f)
  rhs = assemble_rhs_from_iterables(mesh, rhsiter)

  while (error > tol) and (i < max_iter):
    # assemble the LHS of the linear system
    Aiter = stiffness_with_diffusivity_iter(mesh, quadrule)
    Miter = mass_with_reaction_iter_2(mesh, quadrule, u, alpha)
    S = assemble_matrix_from_iterables(mesh, Miter, Aiter)

    # solve the system
    bindices = np.unique(mesh.lines)
    u_new = solve_with_dirichlet_data(S, rhs, bindices, np.zeros_like(bindices))

    # compute error
    error = np.linalg.norm(u - u_new, ord = np.inf)
    errors[i] = error

    # update u
    u = u_new
    i += 1 
  
  print("Number of iterations: ", i)
  print("Final error: ", errors[i-1])
  mesh.tripcolor(u)

# QUESTION 3
def solve_anderson(mesh: Triangulation, quadrule: QuadRule, u0:np.array, tol: float, alpha:float):
  f = lambda x: np.array([100])

  # assemble the RHS of the linear system only once
  rhsiter = poisson_rhs_iter(mesh, quadrule, f)
  rhs = assemble_rhs_from_iterables(mesh, rhsiter)

  def F_anderson(u):
    # assemble the LHS of the linear system
    Aiter = stiffness_with_diffusivity_iter(mesh, quadrule)
    Miter = mass_with_reaction_iter_2(mesh, quadrule, u, alpha)
    S = assemble_matrix_from_iterables(mesh, Miter, Aiter)

    # solve the system
    bindices = np.unique(mesh.lines)
    u_new = solve_with_dirichlet_data(S, rhs, bindices, np.zeros_like(bindices))

    return u_new - u
  
  u_anderson = optimize.anderson(lambda u0: F_anderson(u0), u0, verbose=True, f_tol=tol)

  mesh.tripcolor(u_anderson)

# QUESTION 4
def solve_newton(mesh: Triangulation, quadrule: QuadRule, u0:np.array, tol:float, alpha:float):
  error = tol + 1
  # f = lambda x: np.ones(x.shape).T
  u = u0
  i = 0
  max_iter = 10000
  errors = np.zeros(max_iter)


  while (error > tol) and (i < max_iter):
    # assemble the linear system
    Aiter = stiffness_with_diffusivity_iter(mesh, quadrule)
    Miter = mass_with_reaction_iter_3(mesh, quadrule, u, alpha)
    rhsiter = rhs_newton(mesh=mesh, quadrule=quadrule, alpha=alpha, un=u)
    # rhs_newton_b(mesh=mesh, quadrule=quadrule, alpha=alpha, un=u) + rhs_newton_a(mesh=mesh, quadrule=quadrule, un=u)  

    S = assemble_matrix_from_iterables(mesh, Miter, Aiter)
    rhs = assemble_rhs_from_iterables(mesh, rhsiter)

    # solve the system
    bindices = np.unique(mesh.lines)
    u_new = u + solve_with_dirichlet_data(S, rhs, bindices, np.zeros_like(bindices))

    # compute error
    error = np.linalg.norm(u - u_new, ord = np.inf)
    errors[i] = error

    print(i, error)

    # update u
    u = u_new
    i += 1 
  
  mesh.tripcolor(u)


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
  # mesh.plot()

  n = mesh.points.shape[0]
  # print("Number of vertices: ", n)
  assert n >= n_min, "The number of vertices should be >= 100"

  # define the quadrature formula
  quadrule = seven_point_gauss_6()

  # define the initial guess for u (as a column vector)
  u0 = u0_val * np.ones(n)

  # QUESTION 2: fixed point method
  # solve_fixed_point(mesh, quadrule, u0, tol, alpha)

  # QUESTION 3: Anderson acceleration
  solve_anderson(mesh=mesh, quadrule=quadrule, u0=u0, tol=tol, alpha=alpha)

  # QUESTION 4: Newton scheme
  # solve_newton(mesh=mesh, quadrule=quadrule, u0=u0, tol=tol, alpha=alpha)


if __name__ == '__main__':
  main()