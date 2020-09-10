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

from utils.matrices import assemble_matrix_Ax, matrix_non_singular, pascal_matrix
from utils.stability import create_expected_stability, check_stability


# The number of x variables
nx = 15

# Tested number of parameter variables p
tested_np = [0, 5]

# Tested number of Lagrange multipliers y (i.e., number of rows in A = [Ax Ap])
tested_ny = [4, 8]

# Tested number of Lagrange multipliers z (i.e., number of rows in J = [Jx Jp])
tested_nz = [0, 5]

# Tested number of unstable/fixed basic variables
tested_nbu = [0, 1]

# Tested number of linearly dependent rows in Ax
tested_nl = [0, 1]

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


@mark.parametrize("np"    , tested_np)
@mark.parametrize("ny"    , tested_ny)
@mark.parametrize("nz"    , tested_nz)
@mark.parametrize("nbu"   , tested_nbu)
@mark.parametrize("nl"    , tested_nl)
@mark.parametrize("ifixed", tested_ifixed)
@mark.parametrize("ilower", tested_ilower)
@mark.parametrize("iupper", tested_iupper)
def test_active_stability_checker(np, ny, nz, nbu, nl, ifixed, ilower, iupper):


    # Add the indices of fixed variables to those that have lower and upper bounds
    # since fixed variables are those that have identical lower and upper bounds
    ilower = list(set(ilower + ifixed))
    iupper = list(set(iupper + ifixed))

    # The number of variables with finite lower and upper bounds, and fixed variables
    nlower = len(ilower)
    nupper = len(iupper)
    nfixed = len(ifixed)

    # The total number of variables x, p, y, z
    t = nx + np + ny + nz

    # Assemble the coefficient matrices Ax and Ap
    Ax = assemble_matrix_Ax(ny, nx, nbu, nl, ifixed)
    Ap = linspace(1, ny*np, ny*np).reshape((ny, np))

    # Assemble the coefficient matrix J = [Jx Jp]
    J = pascal_matrix(nz, nx + np)

    # Extract the blocks of J = [Jx Jp]
    Jx = J[:, :nx]
    Jp = J[:, nx:]

    # Create vectors for the lower and upper bounds of the x variables
    xlower = -linspace(1.0, nx, nx) * 100
    xupper =  linspace(1.0, nx, nx) * 100

    # Create vectors for the lower and upper bounds of the p variables
    plower = -linspace(1.0, np, np) * 1000
    pupper =  linspace(1.0, np, np) * 1000

    # Set lower and upper bounds to equal values for fixed variables
    xlower[ifixed] = xupper[ifixed] = linspace(1, nfixed, nfixed) * 10

    # Create vectors x, p, y, z
    x = linspace(1.0, nx, nx)
    p = linspace(1.0, np, np)
    y = linspace(1.0, ny, ny)
    z = linspace(1.0, nz, nz)

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

    # The gradient vector remembering that g + tr(Ax)*y + tr(Jx)*z = s, with s = 0 for stable variables and s != 0 for unstable variables
    g = ones(nx)

    # Ensure g has values for lower/upper unstable variables that make these recognized as unstable indeed
    # Note that:
    #  - x[i] is lower unstable when x[i] == xlower[i] and z[i] > 0, where z = g + tr(Ax)*y + tr(Jx)*z
    #  - x[i] is upper unstable when x[i] == xupper[i] and z[i] < 0, where z = g + tr(Ax)*y + tr(Jx)*z
    g[iunstable_lower] =  inf
    g[iunstable_upper] = -inf

    # Compute the right-hand side vector b remembering that b = Ax*x + Ap*p
    b = Ax @ xlower + Ap @ plower

    # Compute the residual vector h of the nonlinear equality constraints h(x, p) = 0
    h = zeros(nz)

    # The unstabilities of the variables defined as s = g + tr(Ax)*y + tr(Jx)*z where A = [Ax Ap].
    s = g + Ax.T @ y - Jx.T @ z

    # Create a StabilityChecker object
    checker = StabilityChecker(nx, np, ny, nz, Ax, Ap)

    checker.initialize(b, xlower, xupper, plower, pupper)

    checker.update(x, y, z, g, Jx, xlower, xupper)

    # Create a Stability object with expected state
    expected_stability = create_expected_stability(Ax, Ap, x, p, b, s, xlower, xupper, plower, pupper)

    check_stability(checker.stability(), expected_stability)
