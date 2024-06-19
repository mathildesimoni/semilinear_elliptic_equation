from scipy import sparse
from utils.solve import solve_with_dirichlet_data
from typing import Iterable, Callable
import numpy as np

from utils.util import np, _
from utils.quad import QuadRule
from utils.mesh import Triangulation


def shape2D_LFE(quadrule: QuadRule) -> np.ndarray:
  r"""
    Return the shape functions evaluated in the quadrature points
    associated with ``quadrule`` for the three local first order
    lagrangian finite element basis functions (hat functions) over the
    unit triangle.

    Parameters
    ----------
    quadrule: :class: QuadRule with quadrule.simplex_type == 'triangle'

    Returns
    -------
    x: np.ndarray
      evaluation of the three local basis functions in the quad points.
      Has shape (nquadpoints, 3)
  """

  assert quadrule.simplex_type == 'triangle'
  points = quadrule.points

  # points = [x, y], with shape (npoints, 2)
  # shapeF0 = 1 - x - y
  # shapeF1 = x
  # shapeF2 = y
  # so we construct the (npoints, 3) matrix shapeF = [1 - x - y, x, y]
  # simply by concatenating 1 - x - y with points = [x, y] along axis 1
  return np.concatenate([ (1 - points.sum(1)).reshape([-1, 1]), points ], axis=1)

def grad_shape2D_LFE(quadrule: QuadRule) -> np.ndarray:
  r"""
    Return the local gradient of the shape functions evaluated in
    the quadrature points associated with ``quadrule`` for the three local
    first order lagrangian finite element basis functions (hat functions)
    over the unit triangle.

    Parameters
    ----------
    quadrule: :class: `QuadRule` with quadrule.simplex_type == 'triangle'

    Returns
    -------
    x : :class: `np.ndarray`
      evaluation of the three local basis functions in the quad points.
      Has shape (nquadpoints, 3, 2), where the first axis refers to the index
      of the quadrature point, the second axis to the index of the local
      basis function and the third to the component of the gradient.

  """
  assert quadrule.simplex_type == 'triangle'

  # number of quadrature points
  nP, = quadrule.weights.shape
  ones = np.ones((nP,), dtype=float)
  zeros = np.zeros((nP,), dtype=float)
  return np.moveaxis( np.array([ [-ones, -ones],
                                 [ones, zeros],
                                 [zeros, ones] ]), -1, 0)



def assemble_matrix_from_iterables(mesh: Triangulation, *system_matrix_iterables) -> sparse.csr_matrix:
  r""" Assemble sparse matrix from triangulation and system matrix iterables.
      For examples, see end of the script. """

  triangles = mesh.triangles
  ndofs = len(mesh.points)

  A = sparse.lil_matrix((ndofs,)*2)

  for tri, *system_mats in zip(triangles, *system_matrix_iterables):

    # this line is equivalent to
    # for mat in system_mats:
    #   A[np.ix_(*tri,)*2] += mat
    A[np.ix_(*(tri,)*2)] += np.add.reduce(system_mats)

  return A.tocsr()


def assemble_rhs_from_iterables(mesh: Triangulation, *rhs_iterables) -> np.ndarray:
  r""" Assemble right hand side from triangulation and local load vector iterables.
      For examples, see end of the script. """

  triangles = mesh.triangles
  ndofs = len(mesh.points)

  rhs = np.zeros((ndofs,), dtype=float)

  for tri, *local_rhss in zip(triangles, *rhs_iterables):
    rhs[tri] += np.add.reduce(local_rhss)

  return rhs

def mass_with_reaction_iter_Un(mesh: Triangulation, quadrule: QuadRule, un, alpha) -> Iterable:
  r"""
    Iterator for the mass matrix, to be passed into `assemble_matrix_from_iterables`.

    Parameters
    ----------

    mesh : :class:`Triangulation`
      An instantiation of the `Triangulation` class, representing the mesh.
    quadrule : :class: `QuadRule`
      Instantiation of the `QuadRule` class with fields quadrule.points and
      quadrule.weights. quadrule.simplex_type must be 'triangle'.
    un  : :np.array
      Solution from the previous iteration
    alpha : :np.float
      Alpha parameter of the problem

  """

  freact = lambda x: alpha * np.square(x)

  weights = quadrule.weights
  qpoints = quadrule.points
  shapeF = shape2D_LFE(quadrule)

  # loop over all points (a, b, c) per triangle and the correponding
  # Jacobi matrix and measure
  for tri, (a, b, c), BK, detBK in zip(mesh.triangles, mesh.points_iter(), mesh.BK, mesh.detBK):
    un_loc = un[tri] 
    un_qpoints = shapeF @ un_loc 

    # this line is equivalent to
    # outer[i, j] = (weights * shapeF[:, i] * shapeF[:, j] * freact(x)).sum()
    # it's a tad faster because it's vectorised
    outer = (weights[:, _, _] * shapeF[..., _] * shapeF[:, _] * freact(un_qpoints)[:, _, _]).sum(0)
    yield outer * detBK

def mass_with_reaction_iter_dUn(mesh: Triangulation, quadrule: QuadRule, un, alpha) -> Iterable:
  r"""
    Iterator for the mass matrix, to be passed into `assemble_matrix_from_iterables`.

    Parameters
    ----------

    mesh : :class:`Triangulation`
      An instantiation of the `Triangulation` class, representing the mesh.
    quadrule : :class: `QuadRule`
      Instantiation of the `QuadRule` class with fields quadrule.points and
      quadrule.weights. quadrule.simplex_type must be 'triangle'.
    un  : :np.array
      Solution from the previous iteration
    alpha : :np.float
      Alpha parameter of the problem

  """

  # freact not passed => take it to be constant one.
  # freact = lambda x: alpha * np.array([x**2])
  freact = lambda x: 3 * alpha * np.square(x)

  weights = quadrule.weights
  qpoints = quadrule.points
  shapeF = shape2D_LFE(quadrule)

  # loop over all points (a, b, c) per triangle and the correponding
  # Jacobi matrix and measure
  for tri, (a, b, c), BK, detBK in zip(mesh.triangles, mesh.points_iter(), mesh.BK, mesh.detBK):
    un_loc = un[tri] 
    un_qpoints = shapeF @ un_loc 

    # this line is equivalent to
    # outer[i, j] = (weights * shapeF[:, i] * shapeF[:, j] * freact(x)).sum()
    # it's a tad faster because it's vectorised
    outer = (weights[:, _, _] * shapeF[..., _] * shapeF[:, _] * freact(un_qpoints)[:, _, _]).sum(0)
    yield outer * detBK

def stiffness_with_diffusivity_iter(mesh: Triangulation, quadrule: QuadRule, fdiffuse: Callable = None) -> Iterable:
  r"""
    Iterator for the stiffness matrix, to be passed into `assemble_matrix_from_iterables`.

    Parameters
    ----------
    mesh : :class:`Triangulation`
      An instantiation of the `Triangulation` class, representing the mesh.
    quadrule : :class: `QuadRule`
      Instantiation of the `QuadRule` class with fields quadrule.points and
      quadrule.weights. quadrule.simplex_type must be 'triangle'.
    fdiffuse: :class: `Callable`
      Function representing the diffusion term.

  """

  if fdiffuse is None:
    fdiffuse = lambda x: np.array([1])

  weights = quadrule.weights
  qpoints = quadrule.points
  grad_shapeF = grad_shape2D_LFE(quadrule)

  # loop over all points (a, b, c) per triangle and the correponding
  # Jacobi matrix and measure
  for (a, b, c), BK, BKinv, detBK in zip(mesh.points_iter(), mesh.BK, mesh.BKinv, mesh.detBK):

    x = qpoints @ BK.T + a[_]

    # evaluate the diffusivity in the global points.
    fdiffx = fdiffuse(x)

    grad_glob = (BKinv.T[_, _] * grad_shapeF[..., _, :]).sum(-1)
    mat = ((weights * fdiffx)[:, _, _] * (grad_glob[..., _, :] * grad_glob[:, _]).sum(-1)).sum(0) * detBK

    yield mat


def poisson_rhs_iter(mesh: Triangulation, quadrule: QuadRule, f: Callable) -> Iterable:
  r"""
    Iterator for assembling the right-hand side corresponding to
    \int f(x) phi_i dx.

    To be passed into the `assemble_rhs_from_iterables` function.

    Parameters
    ----------

    mesh : :class:`Triangulation`
      An instantiation of the `Triangulation` class, representing the mesh.
    quadrule : :class: `QuadRule`
      Instantiation of the `QuadRule` class with fields quadrule.points and
      quadrule.weights. quadrule.simplex_type must be 'triangle'.
    f : :class: `Callable`
      Function representing the right hand side as a function of the position.
      Must take as input a vector of shape (nquadpoints, 2) and return either
      a vector of shape (nquadpoints,) or (1,).
      The latter means f is constant.
  """

  weights = quadrule.weights
  qpoints = quadrule.points
  shapeF = shape2D_LFE(quadrule)

  for (a, b, c), BK, detBK in zip(mesh.points_iter(), mesh.BK, mesh.detBK):

    # push forward of the local quadpoints (c.f. mass matrix with reaction term).
    x = qpoints @ BK.T + a[_]

    # rhs function f evaluated in the push-forward points
    fx = f(x)

    yield (shapeF * (weights * fx)[:, _]).sum(0) * detBK

def newton_rhs_iter(mesh: Triangulation, quadrule: QuadRule, alpha: float, un: np.array) -> Iterable:

  f = lambda x: alpha * np.power(x, 3) - 100

  weights = quadrule.weights
  qpoints = quadrule.points
  shapeF = shape2D_LFE(quadrule)

  for tri, (a, b, c), BK, detBK in zip(mesh.triangles, mesh.points_iter(), mesh.BK, mesh.detBK):

    un_loc = un[tri] 
    un_qpoints = shapeF @ un_loc 

    yield (shapeF * (weights * f(un_qpoints))[:, _]).sum(0) * detBK