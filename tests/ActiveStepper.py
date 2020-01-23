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

    # Ensure no fixed variable is a lower/upper unstable variable
    jlower = list(set(jlower) - set(jfixed))
    jupper = list(set(jupper) - set(jfixed))

    # Count the number of each variable with fixed value, lower bound, and upper bound
    nfixed = len(jfixed)
    nlower = len(jlower)
    nupper = len(jupper)

    # The total number of rows in W = [A; J]
    m = ml + mn

    # The total number of variables x and y
    t = n + m

    W = assemble_A(m, n, jfixed)  # W = [A; J]

    A = W[:ml, :]
    J = W[ml:, :]

    x = linspace(1, n, n)
    y = linspace(1, m, m)

    xfixed = linspace(1, nfixed, nfixed)
    xlower = eigen.zeros(nlower)
    xupper = eigen.ones(nupper)

    x[jlower] = xlower
    x[jupper] = xupper

    b = A @ x  # *** IMPORTANT *** b = A*x is essential here when A has linearly dependent rows, because it ensures a consistent set of values for vector b (see note in the documentation of SaddlePointSolver class).
    h = zeros(mn)

    g = linspace(1, n, n)
    H = matrix_non_singular(n)

    # Set gradient vector so that lower/upper variables are recognized unstable
    g[jlower] =  float('inf')  # ensure the jlower variables are marked lower unstable (when x[i] == xlower[i] and z[i] > 0, where z = g + tr(A)*y)
    g[jupper] = -float('inf')  # ensure the jupper variables are marked upper unstable (when x[i] == xupper[i] and z[i] < 0, where z = g + tr(A)*y)

    # Ensure off-diagonal entries are zero if Rangespace method is used
    if method == SaddlePointMethod.Rangespace:
        H = eigen.diag(diag(H))

    # Set options for the step calculation
    options = Options()
    options.kkt.method = method

    # Create a stepper calculator
    stepper = ActiveStepper(n, m, A, xlower, xupper, jlower, jupper, jfixed)
    stepper.setOptions(options)
    stepper.decompose(x, y, J, g, H)

    # The solution of the step calculation
    dx = zeros(n)
    dy = zeros(m)
    rx = zeros(n)
    ry = zeros(m)
    z  = zeros(n)

    stepper.solve(x, y, b, h, g, dx, dy, rx, ry, z)

    # Assemble the matrix M = [H, W.T; W, 0] setting rows/cols wrt jfixed to zero
    M = zeros((t, t))
    M[:n, :] = concatenate([H, W.T], axis=1)
    M[n:t, :n] = W

    # Assemble the right-hand side vector of the step calculation
    r = zeros(t)
    r[:n]       = -(g + W.T @ y)
    r[n:t][:ml] = -(A @ x - b)
    r[n:t][ml:] = -(h)

    # Collect the indices of the lower/upper unstable variables and fixed variables
    indices = list(set(jlower) | set(jupper) | set(jfixed))

    # Set rows/cols wrt to unstable and fixed variables to zero (one in the diagonal)
    M[indices, :] = 0.0
    M[:, indices] = 0.0
    M[indices, indices] = 1.0
    r[indices]   = 0.0

    # Calculate r* = M * [dx; dy]
    s = concatenate([dx, dy])
    rstar = M @ s

    assert allclose(rstar, r)
