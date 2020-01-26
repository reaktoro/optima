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
    list(range(n))  # all variables with lower bounds
]

# Tested cases for the indices of variables with upper bounds
tested_iupper = [
    [],
    [1],
    [1, 3, 7, 9],
    list(range(n))  # all variables with upper bounds
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
                   tested_ifixed,
                   tested_ilower,
                   tested_iupper,
                   tested_methods)


@mark.parametrize("args", testdata)
def test_active_stepper(args):

    assemble_A, structure_H, ml, mn, ifixed, ilower, iupper, method = args

    # Add the indices of fixed variables to those that have lower and upper bounds
    # since fixed variables are those that have identical lower and upper bounds
    ilower = list(set(ilower + ifixed))
    iupper = list(set(iupper + ifixed))

    # The number variables with finite lower and upper bounds, and fixed variables
    nlower = len(ilower)
    nupper = len(iupper)
    nfixed = len(ifixed)

    # The total number of rows in W = [A; J]
    m = ml + mn

    # The total number of variables x and y
    t = n + m

    # Assemble the coefficient matrix W = [A; J]
    W = assemble_A(m, n, ifixed)

    # Create references to the A and J blocks in W
    A = W[:ml, :]
    J = W[ml:, :]

    # Create vectors for the lower and upper bounds of the variables
    xlower = full(n, -inf)
    xupper = full(n,  inf)

    # Set lower and upper bounds to negative and positive sequence respectively
    xlower[ilower] = -linspace(1.0, nlower, nlower) * 100
    xupper[iupper] =  linspace(1.0, nupper, nupper) * 100

    # Set lower and upper bounds to equal values for fixed variables
    xlower[ifixed] = xupper[ifixed] = linspace(1, nfixed, nfixed) * 10

    # Create vectors x and y
    x = linspace(1.0, n, n)
    y = linspace(1.0, m, m)

    # Create the Hessian matrix *H*
    H = matrix_non_singular(n)

    # Ensure off-diagonal entries are zero if Rangespace method is used
    if method == SaddlePointMethod.Rangespace:
        H = eigen.diag(diag(H))

    # Auxiliary functions to get first and last of a sequence in a set
    first = lambda seq: [seq[0]] if len(seq) > 0 else []
    last = lambda seq: [seq[-1]] if len(seq) > 0 else []

    # Set first and last variables with lower/upper bounds to be unstable
    iunstable_lower = list(set(first(ilower) + last(ilower)))
    iunstable_upper = list(set(first(iupper) + last(iupper)))
    iunstable = list(set(iunstable_lower + iunstable_upper))

    # Create vector s = (dx, dy) with the expected Newton step values
    s = s_expected = linspace(1.0, t, t)  # Introduce `s` as an alias for `s_expected`

    # Get references to the segments dx and dy in s
    dx = s[:n]
    dy = s[n:]

    # Set to zero the entries in dx corresponding to lower/upper unstable variables
    dx[iunstable] = 0.0

    # Set lower/upper unstable variables in x to their respective lower/upper bounds
    x[iunstable_lower] = xlower[iunstable_lower]
    x[iunstable_upper] = xupper[iunstable_upper]

    # Assemble the matrix M = [H, W.T; W, 0] setting rows/cols wrt ifixed to zero
    M = zeros((t, t))
    M[:n, :n] = H
    M[:n, n:] = W.T
    M[n:, :n] = W

    # Set rows wrt to unstable variables to zero and one in the diagonal entry
    M[iunstable, :] = 0.0
    M[iunstable, iunstable] = 1.0

    # Compute the right-hand side vector r = Ms
    r = r_expected = M @ s  # Introduce `r` as an alias for `r_expected`

    # Get references to rx and ry in r where rx := -(g + tr(W)y) and ry := (ryl, ryn)
    rx = r[:n]
    ry = r[n:]

    # Get references to ryl and ryn in ry where ryl := -(Ax - b) and ryn := -h
    ryl = ry[:ml]
    ryn = ry[ml:]

    # Compute the gradient vector remembering that rx := -(g + tr(W)y) for stable variables, zero otherwise
    g = -(rx + W.T @ y)

    # Ensure g has values for lower/upper unstable variables that make these recognized as unstable indeed
    # Note that:
    #  - x[i] is lower unstable when x[i] == xlower[i] and z[i] > 0, where z = g + tr(W)*y
    #  - x[i] is upper unstable when x[i] == xupper[i] and z[i] < 0, where z = g + tr(W)*y
    g[iunstable_lower] =  inf
    g[iunstable_upper] = -inf

    # Compute the right-hand side vector b remembering that ry = -(Ax - b)
    b = ryl + A @ x

    # Compute the residual vector h of the nonlinear equality constraints h(x) = 0 knowing that ryn := -h
    h = -ryn

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

    # Set options for the Newton step calculation
    options = Options()
    options.kkt.method = method

    # Create a Newton step calculator
    stepper = ActiveStepper(n, m, A)
    stepper.setOptions(options)

    # Initialize, decompose the saddle point matrix, and solve the Newton step
    stepper.initialize(xlower, xupper, iordering)
    stepper.decompose(x, y, g, H, J, xlower, xupper, iordering, nul, nuu)
    stepper.solve(x, y, b, h, g, iordering, dx, dy, rx, ry, z)

    # Compare the actual and expected Newton steps
    s_actual = concatenate([dx, dy])

    assert allclose(M @ s_actual, M @ s_expected)

    # Compare the actual and expected right-hand side vector r of residuals
    r_actual = concatenate([rx, ry])

    assert allclose(r_actual, r_expected)

    # Compare the actual and expected z vector
    z_actual = z
    z_expected = g + W.T @ y

    assert allclose(z_actual, z_expected)

    # Compare the actual and expected number of lower/upper unstable variables
    assert nul.value == [x[i] == xlower[i] and z[i] > 0.0 for i in range(n)].count(True)
    assert nuu.value == [x[i] == xupper[i] and z[i] < 0.0 for i in range(n)].count(True)

    set_printoptions(suppress=True, linewidth=1000, precision=3)
    print()
    print(f"assemble_A  = {assemble_A}")
    print(f"structure_H = {structure_H}")
    print(f"ml          = {ml}")
    print(f"mn          = {mn}")
    print(f"ifixed      = {ifixed}")
    print(f"ilower      = {ilower}")
    print(f"iupper      = {iupper}")
    print(f"method      = {method}")
    print()
    print(f"M = \n{M}")
    print(f"dx(actual)   = {dx}")
    print(f"dx(expected) = {s[:n]}")
    print(f"dy(actual)   = {dy}")
    print(f"dy(expected) = {s[n:]}")
    print(f"rx(actual)   = {rx}")
    print(f"rx(expected) = {r[:n]}")
    print(f"ry(actual)   = {ry}")
    print(f"ry(expected) = {r[n:]}")
    print()
