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
tested_mA = [6, 4]
tested_mJ = [3, 1, 0]

# Combination of all tested cases
testdata = product(tested_matrices_A,
                   tested_structures_H,
                   tested_mA,
                   tested_mJ,
                   tested_jfixed,
                   tested_jlower,
                   tested_jupper,
                   tested_methods)


@mark.parametrize("args", testdata)
def test_active_stepper(args):

    assemble_A, structure_H, mA, mJ, jfixed, jlower, jupper, method = args

    nfixed = len(jfixed)
    nlower = len(jlower)
    nupper = len(jupper)

    m = mA + mJ

    M = assemble_A(m, n, jfixed)

    A = M[:mA, :]
    J = M[mA:, :]

    x = linspace(1, n, n)
    y = linspace(1, m, m)

    xfixed = linspace(1, nfixed, nfixed)
    xlower = eigen.zeros(nlower)
    xupper = eigen.ones(nupper)

    x[jlower] = xlower
    x[jupper] = xupper

    b = A @ x  # *** IMPORTANT *** b = A*x is essential here when A has linearly dependent rows, because it ensures a consistent set of values for vector b (see note in the documentation of SaddlePointSolver class).
    h = zeros(mJ)

    g = linspace(1, n, n)
    H = matrix_non_singular(n)

    g[jlower] =  float('inf')  # ensure the jlower variables are marked lower unstable (when x[i] == xlower[i] and z[i] > 0, where z = g + tr(A)*y)
    g[jupper] = -float('inf')  # ensure the jupper variables are marked upper unstable (when x[i] == xupper[i] and z[i] < 0, where z = g + tr(A)*y)

    if method == SaddlePointMethod.Rangespace:
        H = eigen.diag(diag(H))

    problem = ActiveStepperProblem(
        x,
        y,
        A,
        b,
        h,
        J,
        g,
        H,
        xlower,
        xupper,
        jlower,
        jupper,
        jfixed)

    options = Options()
    options.kkt.method = method

    stepper = ActiveStepper()

    stepper.setOptions(options)
    stepper.decompose(problem)
    stepper.solve(problem)

    M = stepper.matrix(problem).array()
    s = stepper.step().array()
    r = stepper.residual().array()

    rstar = M @ s

    assert allclose(rstar, r)
