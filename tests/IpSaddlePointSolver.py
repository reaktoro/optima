# Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
#
# Copyright (C) 2014-2018 Allan Leal
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from optima import *
from numpy import *
from numpy.linalg import norm
from pytest import approx, mark
from itertools import product

import Canonicalizer


def print_state(M, r, s, m, n):
    set_printoptions(linewidth=1000, precision=10, threshold='nan')
    slu = eigen.solve(M, r)
    print( 'M = \n', M )
    print( 'r        = ', r )
    print( 'x        = ', s[:n] )
    print( 'x(lu)    = ', slu[:n] )
    print( 'x(diff)  = ', abs(s[:n] - slu[:n]) )
    print( 'y        = ', s[n:n + m] )
    print( 'y(lu)    = ', slu[n:n + m] )
    print( 'y(diff)  = ', abs(s[n:n + m] - slu[n:n + m]) )
    print( 'z        = ', s[n + m:n + m + n] )
    print( 'z(lu)    = ', slu[n + m:n + m + n] )
    print( 'z(diff)  = ', abs(s[n + m:n + m + n] - slu[n + m:n + m + n]) )
    print( 'w        = ', s[:n] )
    print( 'w(lu)    = ', slu[:n] )
    print( 'w(diff)  = ', abs(s[:n] - slu[:n]) )
    print( 'res      = ', M.dot(s) - r )
    print( 'res(lu)  = ', M.dot(slu) - r )


# Tested number of variables in (s, l, u, z, w) partitions
tested_dimensions = [
#    m   ns  nl  nu  nz  nw
    (5, 10, 0, 0, 0, 0),
    (5, 8, 2, 0, 0, 0),
    (5, 8, 0, 2, 0, 0),
    (5, 8, 0, 0, 2, 0),
    (5, 8, 0, 0, 0, 2),
]

# Tested cases for the matrix A
tested_matrices_A = Canonicalizer.tested_matrices_A

# Tested cases for the structure of matrix H
tested_structures_H = [
    'dense',
    'diagonal'
]

# Tested cases for the indices of fixed variables
tested_jf = [
    arange(0),
    arange(1),
    array([1, 3, 7, 9])
]

# Tested cases for the saddle point methods
tested_methods = [
    SaddlePointMethod.Fullspace,
    SaddlePointMethod.Nullspace,
    SaddlePointMethod.Rangespace,
    ]

# Combination of all tested cases
testdata = product(tested_dimensions,
                   tested_matrices_A,
                   tested_structures_H,
                   tested_jf,
                   tested_methods
                   )


@mark.parametrize("args", testdata)
def test_ip_saddle_point_solver(args):
    dimensions, assemble_A, structure_H, jf, method = args

    m, ns, nl, nu, nz, nw = dimensions

    n = ns + nl + nu + nz + nw
    t = 3 * n + m

    nf = len(jf)

    A = assemble_A(m, n, nf)
    H = eigen.random(n, n) if structure_H == 'dense' else eigen.random(n)
    Z = eigen.random(n)
    W = eigen.random(n)
    L = eigen.random(n)
    U = eigen.random(n)

    if nl > 0: L[:nl] = 1.0e-3; Z[:nl] = 1.0
    if nu > 0: U[:nu] = 1.0e-3; W[:nu] = 1.0

    if nz > 0: L[nl:nz] = 1.0e-18; Z[nl:nz] = 1.0
    if nw > 0: U[nu:nw] = 1.0e-18; W[nu:nw] = 1.0

    expected = linspace(1, t, t)

    options = SaddlePointOptions()
    options.method = method

    M = eigen.zeros(t, t)
    r = eigen.zeros(t)

    # The left-hand side coefficient matrix
    lhs = IpSaddlePointMatrix(H, A, Z, W, L, U, jf)

    # The dense matrix assembled from lhs
    M = lhs.array()

    # The right-hand side vector
    r = M.dot(expected)
    rhs = IpSaddlePointVector(r, n, m)

    # The solution vector
    s = eigen.zeros(t)
    sol = IpSaddlePointSolution(s, n, m)

    # Solve the interior-poin saddle point problem
    solver = IpSaddlePointSolver()
    solver.setOptions(options)
    solver.initialize(A)
    solver.decompose(lhs)
    solver.solve(rhs, sol)

    # Comment out line below to get further insight of the results when an error happens
#     print_state(M, r, s, m, n)

    # Check the residual of the equation M * s = r
    assert norm(M.dot(s) - r) / norm(r) == approx(0.0)

