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

# The number of variables and number of equality constraints
n = 10
m = 5

# Tested cases for the matrix A
tested_matrices_A = Canonicalizer.tested_matrices_A

# Tested cases for the structure of matrix H
tested_structures_H = [
    'denseH',
    'diagonalH'
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

# Combination of all tested cases
testdata = product(tested_matrices_A,
                   tested_structures_H,
                   tested_jfixed,
                   tested_jlower,
                   tested_jupper,
                   tested_methods)

@mark.parametrize("args", testdata)
def test_stepper(args):

    assemble_A, structure_H, jfixed, jlower, jupper, method = args

    nfixed = len(jfixed)
    nlower = len(jlower)
    nupper = len(jupper)

    t = 3*n + m

    A = assemble_A(m, n, nfixed)

    constraints = Constraints(n)
    constraints.setVariablesWithFixedValues(jfixed)
    constraints.setVariablesWithLowerBounds(jlower)
    constraints.setVariablesWithUpperBounds(jupper)
    constraints.setEqualityConstraintMatrix(A)

    state = State()
    state.x = linspace(1, n, n)
    state.y = linspace(1, m, m)
    state.z = linspace(1, n, n)
    state.w = linspace(1, n, n)

    f = ObjectiveResult()
    f.gradient = linspace(1, n, n)
    f.hessian = eigen.randomSPD(n)

    if method == SaddlePointMethod.Rangespace:
        f.hessian = abs(eigen.diag(eigen.random(n)))

    params = Params()
    params.be = A.dot(state.x)  # *** IMPORTANT *** b = A*x is essential here when A has linearly dependent rows, because it ensures a consistent set of values for vector b (see note in the documentation of SaddlePointSolver class).
    params.xfixed = linspace(1, nfixed, nfixed)
    params.xlower = eigen.zeros(nlower)
    params.xupper = eigen.ones(nupper)

    options = Options()
    options.kkt.method = method

    stepper = Stepper(constraints)
    stepper.setOptions(options)
    stepper.decompose(params, state, f)
    M = stepper.matrix(params, state, f).array()
    expected = linspace(1, t, t)
    rhs = M.dot(expected)

    stepper.solve(params, state, f)

    s = stepper.step().array()
    r = stepper.residual().array()

    assert norm(M.dot(s) - r) / norm(r) == approx(0.0)
