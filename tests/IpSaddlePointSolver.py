# # Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
# #
# # Copyright (C) 2014-2018 Allan Leal
# #
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# # GNU General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with this program. If not, see <http://www.gnu.org/licenses/>.

# from optima import *
# from numpy import *
# from numpy.linalg import norm
# from pytest import approx, mark
# from itertools import product

# from utils.matrices import testing_matrices_W


# def print_state(M, r, s, m, n):
#     slu = eigen.solve(M, r)
#     print( 'M = \n' )
#     for i in range(M.shape[0]):
#         for j in range(M.shape[1]):
#             print(M[i, j], end = ', ')
#         print()
#     print( 'r        = ', r )
#     print( 'x        = ', s[:n] )
#     print( 'x(lu)    = ', slu[:n] )
#     print( 'x(diff)  = ', abs(s[:n] - slu[:n]) )
#     print( 'y        = ', s[n:n + m] )
#     print( 'y(lu)    = ', slu[n:n + m] )
#     print( 'y(diff)  = ', abs(s[n:n + m] - slu[n:n + m]) )
#     print( 'z        = ', s[n + m:n + m + n] )
#     print( 'z(lu)    = ', slu[n + m:n + m + n] )
#     print( 'z(diff)  = ', abs(s[n + m:n + m + n] - slu[n + m:n + m + n]) )
#     print( 'w        = ', s[:n] )
#     print( 'w(lu)    = ', slu[:n] )
#     print( 'w(diff)  = ', abs(s[:n] - slu[:n]) )
#     print( 'res      = ', M.dot(s) - r )
#     print( 'res(lu)  = ', M.dot(slu) - r )


# # Tested number of variables in (s, l, u, z, w) partitions
# tested_dimensions = [
# #    ns  nl  nu  nz  nw
#     (20,  0,  0,  0,  0),
#     (18,  2,  0,  0,  0),
#     (18,  0,  2,  0,  0),
#     (18,  0,  0,  2,  0),
#     (18,  0,  0,  0,  2),
# ]

# # Tested cases for the matrix W
# tested_matrices_W = testing_matrices_W

# # Tested cases for the structure of matrix H
# tested_structures_H = [
#     'dense',
#     'diagonal'
# ]

# # Tested cases for the indices of fixed variables
# tested_jf = [
#     arange(0),
#     arange(1),
#     array([1, 3, 7, 9])
# ]

# # Tested number of rows in matrix Au (upper block of A)
# tested_mu = [6, 4]
# tested_ml = [3, 1, 0]

# # Tested cases for the saddle point methods
# tested_methods = [
#     # SaddlePointMethod.Fullspace,
#     SaddlePointMethod.Nullspace,
#     SaddlePointMethod.Rangespace,
#     ]

# # Combination of all tested cases
# testdata = product(tested_dimensions,
#                    tested_matrices_W,
#                    tested_structures_H,
#                    tested_jf,
#                    tested_mu,
#                    tested_ml,
#                    tested_methods
#                    )


# @mark.parametrize("args", testdata)
# def test_ip_saddle_point_solver(args):
#     dimensions, assemble_W, structure_H, jf, mu, ml, method = args

#     ns, nl, nu, nz, nw = dimensions

#     m = mu + ml

#     n = ns + nl + nu + nz + nw
#     t = 3 * n + m

#     nf = len(jf)

#     jx = list(set(range(n)) - set(jf))  # indices of free variables

#     A = assemble_W(m, n, jf)
#     # print(f"A = \n{A}")
#     # H = eigen.random(n, n)
#     # Z = eigen.random(n)
#     # W = eigen.random(n)
#     # L = eigen.random(n)
#     # U = eigen.random(n)
#     H = eigen.eye(n)
#     Z = eigen.ones(n)
#     W = eigen.ones(n)
#     L = eigen.ones(n)
#     U = eigen.ones(n)

#     Au = A[:mu, :]  # extract the upper block of A
#     Al = A[mu:, :]  # extract the lower block of A

#     if method == SaddlePointMethod.Rangespace:
#         H = eigen.diag(eigen.ones(n))

#     if nl > 0: L[jx[:nl]] = 1.0e-3; Z[jx[:nl]] = 1.0  # jx[:nl] means last nl free variables
#     if nu > 0: U[jx[:nu]] = 1.0e-3; W[jx[:nu]] = 1.0  # jx[:nu] means last nl free variables

#     if nz > 0: L[jx[nl:nz]] = 1.0e-18; Z[jx[nl:nz]] = 1.0  # jx[nl:nz] means free variables in the inverval (nl:nz)
#     if nw > 0: U[jx[nu:nw]] = 1.0e-18; W[jx[nu:nw]] = 1.0  # jx[nu:nw] means free variables in the inverval (nu:nw)

#     expected = linspace(1, t, t)

#     options = SaddlePointOptions()
#     options.method = method

#     M = eigen.zeros(t, t)
#     r = eigen.zeros(t)

#     # The left-hand side coefficient matrix
#     lhs = IpSaddlePointMatrix(H, Au, Al, Z, W, L, U, jf)

#     # The dense matrix assembled from lhs
#     M = lhs.array()

#     # The right-hand side vector
#     r = M.dot(expected)
#     rhs = IpSaddlePointVector(r, n, m)

#     # The solution vector
#     s = eigen.zeros(t)
#     sol = IpSaddlePointSolution(s, n, m)

#     # Solve the interior-poin saddle point problem
#     solver = IpSaddlePointSolver()
#     solver.setOptions(options)
#     solver.decompose(lhs)
#     solver.solve(rhs, sol)

#     # Comment out line below to get further insight of the results when an error happens
#     # print_state(M, r, s, m, n)

#     # Check the residual of the equation M * s = r
#     if norm(M.dot(s) - r) / norm(r) != approx(0.0):
#         set_printoptions(linewidth=1000, threshold=2000)
#         print()
#         print(f"dimensions = {dimensions}")
#         print(f"assemble_W = {assemble_W}")
#         print(f"structure_H = {structure_H}")
#         print(f"jf = {jf}")
#         print(f"mu = {mu}")
#         print(f"ml = {ml}")
#         print(f"method = {method}")
#         print()
#         print(f"A = \n{A}")
#         print_state(M, r, s, m, n)

#     assert norm(M.dot(s) - r) / norm(r) == approx(0.0)

