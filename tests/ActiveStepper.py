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

from utils.matrices import testing_matrices_A, matrix_non_singular


from enum import Enum

class MatrixStructure(Enum):
    Zero = 1
    Diagonal = 2
    Dense = 3


# The number of variables
n = 15

# Tested cases for the matrix A
tested_matrices_A = testing_matrices_A

# Tested cases for the structure of matrix H
tested_structures_H = [
    MatrixStructure.Dense,
    MatrixStructure.Diagonal
]

# Tested cases for the indices of fixed variables
tested_jfixed = [
    arange(0),
    arange(1),
    array([1, 3, 7, 9])
]

# Tested cases for the indices of variables with lower bounds
tested_jlower = [
    arange(0),
    arange(1),
    array([1, 3, 7, 9]),
    arange(n)  # all variables with lower bounds
]

# Tested cases for the indices of variables with upper bounds
tested_jupper = [
    arange(0),
    arange(1),
    array([1, 3, 7, 9]),
    arange(n)  # all variables with upper bounds
]

# Tested cases for the saddle point methods
tested_methods = [
    SaddlePointMethod.Fullspace,
    SaddlePointMethod.Nullspace,
    SaddlePointMethod.Rangespace,
]

# Tested number of rows in matrix A and J
tested_ml = [6, 4]
tested_mn = [3, 1, 0]

# Combination of all tested cases
testdata = product(tested_matrices_A,
                   tested_structures_H,
                   tested_ml,
                   tested_mn,
                   tested_jfixed,
                   tested_jlower,
                   tested_jupper,
                   tested_methods)


@mark.parametrize("args", testdata)
def test_active_stepper(args):

    assemble_A, structure_H, ml, mn, jfixed, jlower, jupper, method = args

    # The number variables with finite lower and upper bounds, and fixed variables
    nlower = len(jlower)
    nupper = len(jupper)
    nfixed = len(jfixed)

    # The total number of rows in W = [A; J]
    m = ml + mn

    # The total number of variables x and y
    t = n + m

    # Assemble the coefficient matrix W = [A; J]
    W = assemble_A(m, n, jfixed)

    # Create references to the A and J blocks in W
    A = W[:ml, :]
    J = W[ml:, :]

    # Create vectors for the lower and upper bounds of the variables
    xlower = full(n, -inf)
    xupper = full(n,  inf)

    # Set lower and upper bounds to negative and positive sequence respectively
    xlower[jlower] = -arange(1.0, nlower)
    xupper[jupper] =  arange(1.0, nupper)

    # Set lower and upper bounds to equal values for fixed variables
    xlower[jfixed] = xupper[jfixed] = arange(1, len(jfixed)) * 10

    # Create vectors x and y
    x = arange(1.0, n)
    y = arange(1.0, m)

    # Create the grandient vector *g* and Hessian matrix *H*
    g = arange(1.0, n)
    H = matrix_non_singular(n)

    # Ensure the first and last variables with lower bounds are recognized as unstable
    # Note: x[i] is lower unstable when x[i] == xlower[i] and z[i] > 0, where z = g + tr(W)*y
    if nlower > 0:
        x[jlower[ 0]] = xlower[jlower[ 0]]
        x[jlower[-1]] = xlower[jlower[-1]]
        g[jlower[ 0]] = inf
        g[jlower[-1]] = inf

    # Ensure the first and last variables with upper bounds are recognized as unstable
    # Note: x[i] is upper unstable when x[i] == xupper[i] and z[i] < 0, where z = g + tr(W)*y
    if nupper > 0:
        x[jupper[ 0]] = xupper[jupper[ 0]]
        x[jupper[-1]] = xupper[jupper[-1]]
        g[jupper[ 0]] = -inf
        g[jupper[-1]] = -inf

    # Ensure all fixed variables (which essentially has identical lower and upper bounds) are unstable
    g[jfixed] = inf

    # Compute the right-hand side vector b in Ax = b
    b = A @ x

    # Create the residual vector of the nonlinear equality constraints *h(x) = 0*
    h = zeros(mn)

    # Create the ordering vector that will order the
    # variables as (*stable*, *lower unstable*, *upper unstable*)
    iordering = arange(n)

    # Create the index numbers that will contain the number
    # of *lower unstable* (nul) and *upper unstable* (nuu) variables
    nul = IndexNumber()
    nuu = IndexNumber()

    # The solution of the Newton step calculation
    dx = zeros(n) # The Newton step for the primal variables *x*.
    dy = zeros(m) # The Newton step for the Lagrange multipliers *y*.
    rx = zeros(n) # The residuals of the first-order optimality conditions.
    ry = zeros(m) # The residuals of the linear/nonlinear feasibility conditions.
    z = zeros(n)  # The *unstabilities* of the variables defined as *z = g + tr(W)y* where *W = [A; J]*.

    # Ensure off-diagonal entries are zero if Rangespace method is used
    if method == SaddlePointMethod.Rangespace:
        H = eigen.diag(diag(H))

    # Set options for the Newton step calculation
    options = Options()
    options.kkt.method = method

    # Create a Newton step calculator
    stepper = ActiveStepper(n, m, A, xlower, xupper, jlower, jupper, jfixed)
    stepper.setOptions(options)

    # Initialize, decompose the saddle point matrix, and solve the Newton step
    stepper.initialize(xlower, xupper, iordering)
    stepper.decompose(x, y, g, H, J, xlower, xupper, iordering, nul, nuu)
    stepper.solve(x, y, b, h, g, iordering, dx, dy, rx, ry, z)

    # Assemble the matrix M = [H, W.T; W, 0] setting rows/cols wrt jfixed to zero
    M = zeros((t, t))
    M[:n, :] = concatenate([H, W.T], axis=1)
    M[n:t, :n] = W

    # Assemble the right-hand side vector of the Newton step calculation
    r = zeros(t)
    r[:n]       = -(g + W.T @ y)
    r[n:t][:ml] = -(A @ x - b)
    r[n:t][ml:] = -(h)

    # Collect the indices of the lower/upper unstable variables and fixed variables
    iu = arange(n)[abs(g) == inf]

    # Set rows/cols wrt to unstable and fixed variables to zero (one in the diagonal)
    M[iu, :]  = 0.0
    M[:, iu]  = 0.0
    M[iu, iu] = 1.0
    r[iu]     = 0.0

    # Calculate r(star) = M * [dx; dy]
    s = concatenate([dx, dy])
    rstar = M @ s

    assert allclose(rstar, r)
