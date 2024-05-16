from scipy import sparse
from solve import solve_with_dirichlet_data
from typing import Iterable, Callable
import numpy as np

from util import np, _
from quad import QuadRule
from mesh import Triangulation


'''
------------------------------------------------------------------------------------------------------------------- 
Problem 2
------------------------------------------------------------------------------------------------------------------- 

'''


def mass_matrix(mesh: Triangulation) -> sparse.csr_matrix:

  # the number of DOFs is equal to the number of mesh vertices / points
  ndofs = len(mesh.points)

  # make empty lil-matrix of shape (ndofs, ndofs)
  # we start with a lil-matrix because it can directly be assigned to
  M = sparse.lil_matrix((ndofs,)*2)

  # the constant block matrix
  # (2, 1, 1), (1, 2, 1), (1, 1, 2)
  mloc = np.ones((3, 3)) + np.eye(3)

  # loop over the triangles
  # for tri, detBK in zip(mesh.triangles, mesh.detBK):
  # produces
  # triangle_indices[0-th triangle], detBK[0-th triangle]
  # triangle_indices[1-st triangle], detBK[1-st triangle]
  # ...
  # triangle_indices[ndofs-1-st triangle], detBK[ndofs-1-st triangle]
  for tri, detBK in zip(mesh.triangles, mesh.detBK):
    # the sub-block of M resulting from slicing-out all rows with indices
    # tri = [i0, i1, i2] and then slicing out all columns with the same indices
    # can be accessed by using the np.ix_ function
    # M[np.ix_(tri, tri)] = the sub-block with row and column indices (i0, i1, i2)
    # np.ix_(tri, tri) is the same as np.ix_(*(tri,)*2)
    M[np.ix_(*(tri,)*2)] += detBK / 24 * mloc

  return M.tocsr()


def stiffness_matrix(mesh: Triangulation) -> sparse.csr_matrix:

  ndofs = len(mesh.points)

  A = sparse.lil_matrix((ndofs,)*2)

  # the local gradients as row-vectors
  # gradshapeF[i] gives the gradient of the i-th local basis function
  gradshapeF = np.array([ [-1, -1],
                          [1, 0],
                          [0, 1] ])

  for tri, detBK, BKinv in zip(mesh.triangles, mesh.detBK, mesh.BKinv):

    # this is equivalent to (BKinv.T @ gradshapeF.T).T (but compacter)
    grad_glob = gradshapeF @ BKinv

    Aloc = np.empty((3, 3), dtype=float)
    for i in range(3):
      for j in range(3):
        Aloc[i, j] = detBK / 2 * (grad_glob[i] * grad_glob[j]).sum()

    # add to the right position in the matrix
    A[np.ix_(*(tri,)*2)] += Aloc

    # this line is the vectorisation of the above which we will learn in the coming weeks
    # _ = np.newaxis
    # A[np.ix_(*(tri,)*2)] += detBK / 2 * (grad[:, _] * grad[_]).sum(-1)

  return A.tocsr()


def load_vector(mesh: Triangulation, F: float = 1.0) -> np.ndarray:

  ndofs = len(mesh.points)

  L = np.zeros((ndofs,), dtype=float)

  for tri, detBK in zip(mesh.triangles, mesh.detBK):
    L[tri] += F / 6 * detBK

  return L


def assemble_neumann_rhs(mesh: Triangulation, mask: np.ndarray, g: float = 1.00) -> np.ndarray:
  mask = np.asarray(mask, dtype=np.bool_)
  mask.shape == mesh.lines.shape[:1]

  local_neumann_load = g / 2 * np.ones(2)

  rhs = np.zeros(len(mesh.points), dtype=float)

  # retain only the boundary edges mesh.lines[i] if mask[i] is True
  neumann_lines = mesh.lines[mask]
  
  # loop over each line [index_of_a, index_of_b] and the corresponding points (a, b)
  for line, (a, b) in zip(neumann_lines, mesh.points[neumann_lines]):
    rhs[line] += ### YOUR CODE HERE

  return rhs


def reaction_diffusion(mesh_size=0.05):
  """
    P1 FEM solution of the reaction-diffusion problem:

      -∆u + u = 1    in  Ω
            u = 0    on ∂Ω_D
         ∂n u = 1    on ∂Ω_N

    where Ω = (0, 1)^2 and ∂Ω_N is the bottom side of Ω.

    parameters
    ----------
    mesh_size: float value 0 < mesh_size < 1 tuning the mesh density.
               Smaller value => denser mesh.

  """

  # create a triangulation of the unit square by passing an array with
  # rows equal to the square's vertices in counter-clockwise direction.
  # The last vertex need not be repeated.
  mesh = Triangulation.from_polygon( np.array([ [0, 0],
                                                [1, 0],
                                                [1, 1],
                                                [0, 1] ]), mesh_size=mesh_size)

  # lines is an integer array of shape (nboundary_edges, 2) containing the indices
  # of the vertices that lie on the boundary edges.
  lines = mesh.lines

  # create a boolean mask of shape mesh.lines[:1] that equals `True`
  # if both associated points' y-values satisfy abs(y) < 1e-10
  neumann_mask = np.abs(mesh.points[lines][..., 1] < 1e-10).all(axis=1)

  # the boundary element that are part of the Dirichlet boundary correspond to the mask
  # that is the negation of `neumann_mask`
  dirichlet_mask = ~neumann_mask

  # plot the mesh
  mesh.plot()

  # make mass matrix
  M = mass_matrix(mesh)

  # make stiffness matrix
  A = stiffness_matrix(mesh)

  # assemble the load vector
  rhs = load_vector(mesh, F=1) + assemble_neumann_rhs(mesh, neumann_mask, g=1)

  # the boundary vertices are the unique indices of the mesh's boundary edges restricted to the Dirichlet boundary
  bindices = np.unique(lines[dirichlet_mask])

  # use the `solve_with_dirichlet_data` method to solve the system under the boundary condition
  solution = solve_with_dirichlet_data(A + M, rhs, bindices, np.zeros_like(bindices))

  mesh.tripcolor(solution)

'''
------------------------------------------------------------------------------------------------------------------- 
Problem 3 
------------------------------------------------------------------------------------------------------------------- 

'''
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

      Example: the gradient of shape (2,) of the 2nd local basis function
               in the third quadrature point is given by x[2, 1, :]
               (0-based indexing).
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

  r"""
    Iterator for the mass matrix, to be passed into `assemble_matrix_from_iterables`.

    Parameters
    ---------- 

    mesh : :class:`Triangulation`
      An instantiation of the `Triangulation` class, representing the mesh.
    quadrule : :class: `QuadRule`
      Instantiation of the `QuadRule` class with fields quadrule.points and
      quadrule.weights. quadrule.simplex_type must be 'triangle'.
    freact: :class: `Callable`
      Function representing the reaction term. Must take as argument a single
      array of shape quadrule.points.shape and return a :class: `np.ndarray`
      object either of shape arr.shape == quadrule.weights.shape or
                             arr.shape == (1,)
      The latter usually means that freact is constant.

    Example
    -------
    For an example, see the end of the script.
  """

  # freact not passed => take it to be constant one.
  if freact is None:
    freact = lambda x: np.array([1])

  weights = quadrule.weights
  qpoints = quadrule.points
  shapeF = shape2D_LFE(quadrule)

  # loop over all points (a, b, c) per triangle and the correponding
  # Jacobi matrix and measure
  for (a, b, c), BK, detBK in zip(mesh.points_iter(), mesh.BK, mesh.detBK):

    # define the global points by pushing forward the local quadrature points
    # from the reference element onto the current triangle
    x = qpoints @ BK.T + a[_]

    # this line is equivalent to
    # outer[i, j] = (weights * shapeF[:, i] * shapeF[:, j] * freact(x)).sum()
    # it's a tad faster because it's vectorised
    outer = (weights[:, _, _] * shapeF[..., _] * shapeF[:, _] * freact(x)[:, _, _]).sum(0)
    yield outer * detBK


def stiffness_with_diffusivity_iter(mesh: Triangulation, quadrule: QuadRule, fdiffuse: Callable = None) -> Iterable:
  r"""
    Iterator for the stiffness matrix, to be passed into `assemble_matrix_from_iterables`.

    Parameters
    ----------

    Exactly the same as in `mass_with_reaction_iter`.
    freact -> fdiffuse and has to be implemented in the exact same way.

    Example
    -------
    For an example, see the end of the script.
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

    # below an implementation using two for loops

    r"""
      mat = np.zeros((3, 3), dtype=float)

      for i in range(3):
        for j in range(i, 3):
          # y = grad_shapeF[:, i] is of shape (nquadpoints, 2) and wherein
          # y[j, k] represents the k-th component of \hat{\nabla} phi_i on the
          # j-th quadrature point.
          # The integral of \nabla phi_i \cdot \nabla phi_j is given by
          # \int BK^{-T} @ (\hat{\nabla} \phi_i) \cdot BK^{-T} @ (\hat{\nabla} phi_j) detBK dxi
          Gi = grad_shapeF[:, i] @ BKinv
          Gj = grad_shapeF[:, j] @ BKinv
          mat[i, j] = (weights * fdiffx * (Gi * Gj).sum(1)).sum() * detBK

      # add strictly upper triangular part transposed to mat
      mat += np.triu(mat, k=1).T
    """

    # these two lines are equivalent to all of the above
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


def load_vector(mesh: Triangulation, F: float = 1.0) -> np.ndarray:

  ndofs = len(mesh.points)

  L = np.zeros((ndofs,), dtype=float)

  for tri, detBK in zip(mesh.triangles, mesh.detBK):
    L[tri] += F / 6 * detBK

  return L

def compute_H1_norm_difference(mesh, quadrule, solution_approx, f, df, a0=1, a1=1):
  r"""
    Compute the H1(Ω)-norm of the difference between the approximate solution,
    characterised by its weights with respect to a P1-FEM basis defined on the
    mesh, and a C^{\infty}(Ω) function f. The function f typically represents
    the exact solution of a PDE problem.

    Parameters
    ----------

    mesh : :class: `Triangulation`
      The mesh.
    quadrule : :class: `QuadRule`
      A quadrature rule with quadrule.simplex_type == 'triangle'.
    solution_approx: :class: `np.ndarray`
      Vector of weights representing the approximate solution with respect
      to the canonical P1-FEM basis associated with `mesh`.
    f : :class: `Callable`
      Function returning the the evaluation of the exact solution in the physical
      coordinates x.
      Given x of shape (nquadpoints,), must return an array of shape (nquadpoints,)
      or of shape (1,).
    df: :class: `Callable`
      Function representing the gradient of f in Ω.
      Given the phyiscal coordinates x of shape (mquadpoints,) must return
      array of shape (nquadpoints, 2), (nquadpoints, 1) (1, 2) or (1, 1).
      Returning an array of shape (nquadpoints, 1) means that the x and y
      derivatives are equal. Shape (1, 2) means that dx and dy are constant but
      unequal. (1, 1) means they are constant and equal.
    a0, a1: `float`
      The weight of the L^2 and semi-H^1 part in the H^1-type norm.
      Here, (a0, a1) = (1, 0) is the L^2-norm while (a0, a1) = (0, 1) is the H^1 semi-norm
      while (a0, a1) = (1, 1) is the ordinary H^1-norm.
  """

  assert solution_approx.shape == mesh.points.shape[:1]

  weights = quadrule.weights
  qpoints = quadrule.points

  grad_shapeF = grad_shape2D_LFE(quadrule)
  shapeF = shape2D_LFE(quadrule)

  norm_squared = 0

  for tri, (a, b, c), BK, BKinv, detBK in zip( mesh.triangles, mesh.points_iter(), mesh.BK, mesh.BKinv, mesh.detBK ):

    local_weights = solution_approx[tri]
    x = qpoints @ BK.T + a[_]
    fx = f(x)
    dfx = df(x)

    u = shapeF @ local_weights
    grad = (grad_shapeF * local_weights[_, :, _]).sum(1) @ BKinv

    norm_squared += (weights * (a1 * ((grad - dfx)**2).sum(1) + a0 * (u - fx)**2)).sum() * detBK

  return np.sqrt(norm_squared)

def solve_problem_3(mesh_size=0.01):
  r"""
    P1 FEM solution of the reaction-diffusion problem:

    -\nabla \cdot (a(x, y) \nabla u) + r(x, y) u = f(x, y)  in  Ω
                                            ∂n u = 0        on ∂Ω

    where Ω = (0, 1)^2.

    Here, a(x, y) = 1 + x, r(x, y) = 4π^2(1 + x)
    and f(x, y) = 2π cos(2πy)[sin(2πx) + 6π(1+x)cos(2πx)]

    parameters
    ----------
    mesh_size : `float`
      Numeric value 0 < mesh_size < 1 tuning the mesh density.
      Smaller value => denser mesh.
  """

  from quad import seven_point_gauss_6
  from scipy.sparse import linalg as splinalg

  # make the square domain with mesh size `mesh_size`
  square = np.array([ [0, 0],
                      [1, 0],
                      [1, 1],
                      [0, 1] ])
  mesh = Triangulation.from_polygon(square, mesh_size=mesh_size)

  quadrule = seven_point_gauss_6()

  # define the diffusivity and reactivity as a function of x with
  # shape (nquadpoints, 2)
  fdiffuse = lambda x: 1 + x[:, 0]
  freact = lambda x: 4 * np.pi**2 * (1 + x[:, 0])

  def f(x: np.ndarray) -> np.ndarray:
    x, y = x.T
    return 2 * np.pi * np.cos(2 * np.pi * y) * \
           (np.sin(2 * np.pi * x) + 6 * np.pi * (1 + x) * np.cos(2 * np.pi * x))

  Aiter = stiffness_with_diffusivity_iter(mesh, quadrule, fdiffuse=fdiffuse)
  Miter = mass_with_reaction_iter(mesh, quadrule, freact=freact)
  rhsiter = poisson_rhs_iter(mesh, quadrule, f)

  S = assemble_matrix_from_iterables(mesh, Miter, Aiter)
  rhs = assemble_rhs_from_iterables(mesh, rhsiter)

  solution_approx = splinalg.spsolve(S, rhs)

  mesh.tripcolor(solution_approx)

  exact = lambda x: np.cos(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])
  dexact = lambda x: np.stack([ -2 * np.pi * np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1]),
                                -2 * np.pi * np.cos(2 * np.pi * x[:, 0]) * np.sin(2 * np.pi * x[:, 1]) ], axis=1)

  dnorm = compute_H1_norm_difference(mesh, quadrule, solution_approx, exact, dexact)

  print('The H1-norm of the difference between the approximate and the exact solution equals {:.6}.'.format(dnorm))



