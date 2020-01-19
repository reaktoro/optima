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

from utils.matrices import testing_matrix_structures

# The number of variables and number of equality constraints
n = 15
m = 5

# Tested cases for the matrix A
tested_matrices_A = testing_matrix_structures

# Tested cases for the structure of matrix H
tested_structures_H = [
    'dense',
    'diagonal'
]

# Tested cases for the indices of fixed variables
tested_jfixed = [
    arange(0),
    array([0]),
    array([0, 1, 2])
]

# Tested cases for the indices of variables with lower bounds
tested_jlower = [
    arange(0),
    array([0]),
    array([3, 4, 5])
]

# Tested cases for the indices of variables with upper bounds
tested_jupper = [
    arange(0),
    array([0]),
    array([6, 7, 8])
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


def objective_fn(x, res):
    res.f = sum((x - 0.5) ** 2)
    res.g = 2.0 * (x - 0.5)
    res.H = 2.0 * eye(len(x))


def constraint_fn(x, res):
    pass


@mark.parametrize("args", testdata)
def test_basic_solver(args):

    assemble_A, structure_H, jfixed, jlower, jupper, method = args

    nlower = len(jlower)
    nupper = len(jupper)
    nfixed = len(jfixed)

    A = assemble_A(m, n, jfixed)

    problem = BasicProblem()

    # Set the dimension information of the problem
    problem.dims.x = n
    problem.dims.b = m
    problem.dims.h = 0
    problem.dims.xlower = nlower
    problem.dims.xupper = nupper
    problem.dims.xfixed = nfixed

    # Set the constraint information of the problem
    problem.constraints.A = A
    # problem.constraints.h = constraint_fn
    problem.constraints.ilower = jlower
    problem.constraints.iupper = jupper
    problem.constraints.ifixed = jfixed

    # Set the objective function of the problem
    problem.objective = objective_fn

    y_expected = linspace(1, m, m)

    z_expected = zeros(n)
    z_expected[jlower] = +1.0  # Slack variables z are +1 at active lower bounds

    w_expected = zeros(n)
    w_expected[jupper] = -1.0  # Slack variables w are -1 at active upper bounds

    jx = list(set(range(n)) - set(jfixed))

    Ax = A[:, jx]  # The columns of A corresponding to free variables
    Atx = array(transpose(Ax))
    zx = array(z_expected[jx])  # The rows of z corresponding to free variables
    wx = array(w_expected[jx])  # The rows of w corresponding to free variables
    gx = zx + wx - Atx.dot(y_expected)
    xx = 0.5 * (gx + 1.0)

    x_expected = zeros(n)
    x_expected[jx] = xx
    x_expected[jfixed] = linspace(1, nfixed, nfixed)

    params = BasicParams()
    params.b      = A.dot(x_expected)  # Use expected values of x to compute b
    params.xfixed = x_expected[jfixed] # Use expected values of x for the fixed values
    params.xlower = x_expected[jlower] # Use expected values of x for the lower bounds
    params.xupper = x_expected[jupper] # Use expected values of x for the upper bounds

    state = BasicState(n, m)

    options = Options()
    # options.output.active = True
    options.kkt.method = method

    solver = BasicSolver(problem)
    solver.setOptions(options)
    res = solver.solve(params, state)

#     print state.x

    if not res.succeeded:
        set_printoptions(linewidth=100000)
        print(f"A = \n{A}")

        state = BasicState(n, m)

        options = Options()
        options.output.active = True
        options.kkt.method = method

        solver = BasicSolver(problem)
        solver.setOptions(options)
        res = solver.solve(params, state)

    assert res.succeeded


#
# def test_basic_solver():
#
#     set_printoptions(linewidth=1000)
#
#     A = eigen.random(m, n)
#
# #     jfixed = []
# #     jlower = []
# #     jupper = [1,2, 3, 4]
#
# #     jfixed = []
# #     jlower = []
# #     jlower = [0, 1, 2]
# #     jupper = [4, 5, 6]
#
#     jfixed = [9]
#     jlower = [1, 2, 3]
#     jupper = [4, 6]
#
# #     jlower = [2, 5, 7, 9]
# #     jupper = [1, 3, 6]
#
#     nfixed = len(jfixed)
#     nlower = len(jlower)
#     nupper = len(jupper)
#
#     structure = Structure(n)
#     structure.setVariablesWithFixedValues(jfixed)
#     structure.setVariablesWithLowerBounds(jlower)
#     structure.setVariablesWithUpperBounds(jupper)
#     structure.A = A
#
#     print 'jlower =', jlower
#     print 'structure.variablesWithLowerBounds() =', structure.variablesWithLowerBounds()
#
# #     factor = 10 ** ceil(log10(n))  # Used to have all x between 0 and 1 no matter the number of variables
# # x_expected = linspace(1, n, n) / factor  # Variables x have values
# # between 0 and 1
#     y_expected = linspace(1, m, m)
#     z_expected = zeros(n)
#     w_expected = zeros(n)
#
# # x_expected[jfixed] = 10 * x_expected[jfixed]  # Fixed variables x have values
# # greater than one
#
# #     x_expected[jlower] = 0.0  # Variables x active at lower bounds are zero
# #     x_expected[jupper] = 1.0  # Variables x active at upper bounds are one
#
#     z_expected[jlower] = 1.0  # Slack variables z are one at active lower bounds
#     w_expected[jupper] = 1.0  # Slack variables w are one at active upper bounds
#
#     jx = list(set(range(n)) - set(jfixed))  # The indices of the free variables
#
#     Ax = A[:, jx]  # The columns of A corresponding to free variables
#     Atx = array(transpose(Ax))
#     zx = array(z_expected[jx])  # The rows of z corresponding to free variables
#     wx = array(w_expected[jx])  # The rows of w corresponding to free variables
#     gx = zx - wx - Atx.dot(y_expected)
#     xx = 0.5 * (gx + 1.0)
#
#     x_expected = zeros(n)
#     x_expected[jx] = xx
#     x_expected[jfixed] = linspace(1, nfixed, nfixed)
#
#     params = Params()
#     params.be = A.dot(x_expected)  # Use expected values of x to compute b
#     params.xfixed = x_expected[jfixed]
#     params.xlower = x_expected[jlower]
#     params.xupper = x_expected[jupper]
#     params.objective = objective
#
#     print 'x_expected =', x_expected
#     print 'y_expected =', y_expected
#     print 'z_expected =', z_expected
#     print 'w_expected =', w_expected
#     print 'xlower     =', params.xlower
#     print 'xupper     =', params.xupper
#
#     state = State()
# #     state.x = 0.5 * eigen.ones(n)
#
#     options = Options()
#     options.output.active = True
#     options.max_iterations = 25
# #     options.kkt.method = method
#     options.kkt.method = SaddlePointMethod.Rangespace
#
#     solver = Solver(structure)
#     solver.setOptions(options)
#     res = solver.solve(params, state)
#
#     print 'x(calculated) =', state.x
#     print 'x(expected)   =', x_expected
#     print
#     print 'y(calculated) =', state.y
#     print 'y(eypected)   =', y_expected
#     print
#     print 'z(calculated) =', state.z
#     print 'z(expected)   =', z_expected
#     print
#     print 'w(calculated) =', state.w
#     print 'w(expected)   =', w_expected
#     print
#     print 'x(diff)       =', abs(state.x - x_expected)
#     print 'y(diff)       =', abs(state.y - y_expected)
#     print 'z(diff)       =', abs(state.z - z_expected)
#     print 'w(diff)       =', abs(state.w - w_expected)
#     print
#     print 'norm(x(diff)) =', norm(abs(state.x - x_expected))
#     print 'norm(y(diff)) =', norm(abs(state.y - y_expected))
#     print 'norm(z(diff)) =', norm(abs(state.z - z_expected))
#     print 'norm(w(diff)) =', norm(abs(state.w - w_expected))
#
#     assert res.succeeded
