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
from pytest import approx, mark
from itertools import product

from utils.matrices import testing_matrices_W, matrix_non_singular
from utils.stability import create_expected_stability, check_stability


# The number of x variables
nx = 15

# Tested number of parameter variables p
tested_np = [0, 5]

# Tested cases for the matrix W
tested_matrices_W = testing_matrices_W

# Tested cases for the indices of fixed variables
tested_ifixed = [
    [],
    [1],
    [1, 3, 7, 9]
]

# Tested cases for the indices of variables with lower bounds
tested_ilower = [
    [],
    [1],
    [1, 3, 7, 9],
    list(range(nx))  # all variables with lower bounds
]

# Tested cases for the indices of variables with upper bounds
tested_iupper = [
    [],
    [1],
    [1, 3, 7, 9],
    list(range(nx))  # all variables with upper bounds
]

# Tested number of rows in matrix A and J
tested_ml = [6, 4]
tested_mn = [3, 1, 0]

# Combination of all tested cases
testdata = product(tested_matrices_W,
                   tested_np,
                   tested_ml,
                   tested_mn,
                   tested_ifixed,
                   tested_ilower,
                   tested_iupper)


@mark.parametrize("args", testdata)
def test_active_stability_checker(args):

    assemble_W, np, ml, mn, ifixed, ilower, iupper = args

    # Add the indices of fixed variables to those that have lower and upper bounds
    # since fixed variables are those that have identical lower and upper bounds
    ilower = list(set(ilower + ifixed))
    iupper = list(set(iupper + ifixed))

    # The number of variables with finite lower and upper bounds, and fixed variables
    nlower = len(ilower)
    nupper = len(iupper)
    nfixed = len(ifixed)

    # The total number of rows in W = [Ax Ap; Jx Jp]
    m = ml + mn

    # The total number of variables x, p, y
    t = nx + np + m

    # Assemble the coefficient matrix W = [Ax Ap; Jx Jp]
    W = assemble_W(m, nx + np, ifixed)

    # Extract the blocks of W = [Wx; Wp] = [Ax Ap; Jx Jp] = [Ax Ap; Jx Jp]
    Wx = W[:, :nx]
    Wp = W[:, nx:]

    Ax = Wx[:ml, :]
    Ap = Wp[:ml, :]

    Jx = Wx[ml:, :]
    Jp = Wp[ml:, :]

    # Create vectors for the lower and upper bounds of the x variables
    xlower = -linspace(1.0, nx, nx) * 100
    xupper =  linspace(1.0, nx, nx) * 100

    # Create vectors for the lower and upper bounds of the p variables
    plower = -linspace(1.0, np, np) * 1000
    pupper =  linspace(1.0, np, np) * 1000

    # Set lower and upper bounds to equal values for fixed variables
    xlower[ifixed] = xupper[ifixed] = linspace(1, nfixed, nfixed) * 10

    # Create vectors x, p, y
    x = linspace(1.0, nx, nx)
    p = linspace(1.0, np, np)
    y = linspace(1.0, m, m)

    # Auxiliary functions to get head and tail of a sequence in a set
    head = lambda seq: [seq[ 0]] if len(seq) > 0 else []
    tail = lambda seq: [seq[-1]] if len(seq) > 0 else []

    # Set head and tail variables with lower/upper bounds to be unstable as well as all fixed variables
    iunstable_lower = list(set(head(ilower) + tail(ilower) + ifixed))
    iunstable_upper = list(set(head(iupper) + tail(iupper) + ifixed))
    iunstable = list(set(iunstable_lower + iunstable_upper))
    istable = list(set(range(nx)) - set(iunstable))

    # The number of unstable variables
    nu = len(iunstable)

    # Set lower/upper unstable variables in x to their respective lower/upper bounds
    x[iunstable_lower] = xlower[iunstable_lower]
    x[iunstable_upper] = xupper[iunstable_upper]

    # The gradient vector remembering that g + tr(W)y = z, with z = 0 for stable variables and z != 0 for unstable variables
    g = ones(nx)

    # Ensure g has values for lower/upper unstable variables that make these recognized as unstable indeed
    # Note that:
    #  - x[i] is lower unstable when x[i] == xlower[i] and z[i] > 0, where z = g + tr(W)*y
    #  - x[i] is upper unstable when x[i] == xupper[i] and z[i] < 0, where z = g + tr(W)*y
    g[iunstable_lower] =  inf
    g[iunstable_upper] = -inf

    # Compute the right-hand side vector b remembering that b = Ax*x + Ap*p
    b = Ax @ xlower + Ap @ plower

    # Compute the residual vector h of the nonlinear equality constraints h(x, p) = 0
    h = zeros(mn)

    # The *unstabilities* of the variables defined as *z = g + tr(W)y* where *W = [Ax Ap; Jx Jp]*.
    z = g + Wx.T @ y

    # Create a StabilityChecker object
    checker = StabilityChecker(nx, np, m, Ax, Ap)

    checker.initialize(b, xlower, xupper, plower, pupper)

    checker.update(x, y, g, Jx, xlower, xupper)

    # Create a Stability object with expected state
    expected_stability = create_expected_stability(Ax, x, b, z, xlower, xupper)

    check_stability(checker.stability(), expected_stability)
