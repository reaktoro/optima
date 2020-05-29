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

from utils.matrices import testing_matrices_W, matrix_non_singular

# The number of variables and number of equality constraints
n = 15
m = 5

# Tested cases for the matrix W
tested_matrices_W = testing_matrices_W

# Tested cases for the structure of matrix H
tested_structures_H = [
    'dense',
    'diagonal'
]

# Tested cases for the indices of fixed variables
tested_ifixed = [
    [],
    [0],
    [0, 1, 2]
]

# Tested cases for the indices of variables with lower bounds
tested_ilower = [
    [],
    [0],
    [3, 4, 5]
]

# Tested cases for the indices of variables with upper bounds
tested_iupper = [
    [],
    [0],
    [6, 7, 8]
]

# Tested cases for the saddle point methods
tested_methods = [
    SaddlePointMethod.Fullspace,
    SaddlePointMethod.Nullspace,
    SaddlePointMethod.Rangespace,
]

# Combination of all tested cases
testdata = product(tested_matrices_W,
                   tested_structures_H,
                   tested_ifixed,
                   tested_ilower,
                   tested_iupper,
                   tested_methods)


def create_objective_fn(H, c):
    def fn(x, res):
        res.f = 0.5 * (x.T @ H @ x) + c.T @ x
        res.g = H @ x + c
        res.H = H
    return fn


def create_constraint_fn():
    def fn(x, res):
        pass
    return fn


@mark.parametrize("args", testdata)
def test_basic_solver(args):

    assemble_W, structure_H, ifixed, ilower, iupper, method = args

    # Add the indices of fixed variables to those that have lower and upper bounds
    # since fixed variables are those that have identical lower and upper bounds
    ilower = list(set(ilower + ifixed))
    iupper = list(set(iupper + ifixed))

    # The number variables with finite lower and upper bounds, and fixed variables
    nlower = len(ilower)
    nupper = len(iupper)
    nfixed = len(ifixed)

    # The total number of variables x and y
    t = n + m

    # Assemble the coefficient matrix W = [A; J]
    W = assemble_W(m, n, ifixed)

    # For the moment, set A in Ax = B to W
    A = W

    # Create vectors for the lower and upper bounds of the variables
    xlower = full(n, -inf)
    xupper = full(n,  inf)

    # Set lower and upper bounds to negative and positive sequence respectively
    xlower[ilower] = -linspace(1.0, nlower, nlower) * 100
    xupper[iupper] =  linspace(1.0, nupper, nupper) * 100

    # Set lower and upper bounds to equal values for fixed variables
    xlower[ifixed] = xupper[ifixed] = linspace(1, nfixed, nfixed) * 1000

    # Auxiliary functions to get head and tail of a sequence in a list (return empty list if empty sequence)
    head = lambda seq: [seq[ 0]] if len(seq) > 0 else []
    tail = lambda seq: [seq[-1]] if len(seq) > 0 else []

    # Set head and tail variables with lower/upper bounds to be unstable as well as all fixed variables
    iunstable_lower = list(set(head(ilower) + tail(ilower) + ifixed))
    iunstable_upper = list(set(head(iupper) + tail(iupper) + ifixed))
    iunstable = list(set(iunstable_lower + iunstable_upper))

    # Create vector s = (x, y) with the expected solution of the optimization problem
    s = linspace(1.0, t, t)

    # Get references to the subvectors x and y in s
    x = s[:n]
    y = s[n:]

    # Set lower/upper unstable variables in x to their respective lower/upper bounds
    x[iunstable_lower] = xlower[iunstable_lower]
    x[iunstable_upper] = xupper[iunstable_upper]

    # Create the expected vector z = g + tr(A)yl + tr(J)yn
    z = zeros(n)
    z[iunstable_lower] =  123  # lower unstable variables have positive value for z
    z[iunstable_upper] = -123  # upper unstable variables have negative value for z

    # Create the Hessian matrix *H*
    H = matrix_non_singular(n)

    # Ensure off-diagonal entries are zero if Rangespace method is used
    if method == SaddlePointMethod.Rangespace:
        H = diag(diag(H))

    # Zero out rows and columns in H corresponding to fixed variables for the sake of computing consistent c vector below
    H[ifixed, :] = H[:, ifixed] = 0.0

    # Compute the expected gradient vector at the solution using z = g + tr(A)yl + tr(J)yn
    g = z - A.T @ y

    # Compute the c vector in f(x) = 1/2 tr(x)Hx + tr(c)x using g = H @ x + c
    c = g - H @ x

    # Create the objective function with assembled H and c
    obj = create_objective_fn(H, c)

    # Create the nonlinear equality constraint function h(x)
    h = create_constraint_fn()

    # Compute vector b in Ax = b
    b = A @ x

    # Create the ordering vector that will order the
    # variables as (*stable*, *lower unstable*, *upper unstable*)
    iordering = arange(n)

    # Create the index numbers that will contain the number
    # of *lower unstable* (nul) and *upper unstable* (nuu) variables
    nul = IndexNumber()
    nuu = IndexNumber()

    # Keep references to current x, y, z as they are the expected solution
    x_expected = x
    y_expected = y
    z_expected = z

    # Create vectors for the solution of the optimization problem
    x = zeros(n)
    y = zeros(m)
    z = zeros(n)

    # Create the options for the optimization calculation
    options = Options()
    options.output.active = False
    options.kkt.method = method

    # Solve the optimization problem
    solver = BasicSolver(n, m, A)
    solver.setOptions(options)

    res = solver.solve(obj, h, b, xlower, xupper, x, y, z, iordering, nul, nuu)

    if not res.succeeded:

        # set_printoptions(linewidth=100000, formatter={'float': '{: 0.3f}'.format})
        set_printoptions(linewidth=100000, precision=6, suppress=True)
        print()
        # print(f"H = \n{H}\n")
        # print(f"A = \n{A}\n")
        print(f"x(actual)   = {x}")
        print(f"x(expected) = {x_expected}")
        print(f"x(diff) = {abs(x - x_expected)}")
        print(f"y(actual)   = {y}")
        print(f"y(expected) = {y_expected}")
        print(f"y(diff) = {abs(y - y_expected)}")
        print(f"z(actual)   = {z}")
        print(f"z(expected) = {z_expected}")
        print(f"z(diff) = {abs(z - z_expected)}")

    assert res.succeeded
