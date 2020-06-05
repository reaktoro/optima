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
from utils.stability import create_expected_stability, check_stability


from enum import Enum

class MatrixStructure(Enum):
    Zero = 1
    Diagonal = 2
    Dense = 3


# The number of variables
n = 15

# Tested cases for the matrix W
tested_matrices_W = testing_matrices_W

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
testdata = product(tested_matrices_W,
                   tested_structures_H,
                   tested_ml,
                   tested_mn,
                   tested_ifixed,
                   tested_ilower,
                   tested_iupper,
                   tested_methods)


@mark.parametrize("args", testdata)
def test_active_stepper(args):

    assemble_W, structure_H, ml, mn, ifixed, ilower, iupper, method = args

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
    W = assemble_W(m, n, ifixed)

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
        H = diag(diag(H))

    # Auxiliary functions to get head and tail of a sequence in a set
    head = lambda seq: [seq[ 0]] if len(seq) > 0 else []
    tail = lambda seq: [seq[-1]] if len(seq) > 0 else []

    # Set head and tail variables with lower/upper bounds to be unstable as well as all fixed variables
    iunstable_lower = list(set(head(ilower) + tail(ilower) + ifixed))
    iunstable_upper = list(set(head(iupper) + tail(iupper) + ifixed))
    iunstable = list(set(iunstable_lower + iunstable_upper))
    istable = list(set(range(n)) - set(iunstable))

    # The number of unstable variables
    nu = len(iunstable)

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

    # The solution of the Newton step calculation
    dx = zeros(n) # The Newton step for the primal variables *x*.
    dy = zeros(m) # The Newton step for the Lagrange multipliers *y*.
    rx = zeros(n) # The residuals of the first-order optimality conditions.
    ry = zeros(m) # The residuals of the linear/nonlinear feasibility conditions.
    z = zeros(n)  # The *unstabilities* of the variables defined as *z = g + tr(W)y* where *W = [A; J]*.

    # Create the stability state of the variables
    stability = Stability()

    # Set options for the Newton step calculation
    options = Options()
    options.kkt.method = method

    # Create a Newton step calculator
    stepper = Stepper(n, m, A)
    stepper.setOptions(options)

    # Initialize, decompose the saddle point matrix, and solve the Newton step
    stepper.initialize(b, xlower, xupper, x, stability)
    stepper.decompose(x, y, g, H, J, xlower, xupper, stability)
    stepper.solve(x, y, b, h, g, H, stability, dx, dy, rx, ry, z)

    # Compute the sensitivity derivatives of the optimal solution
    np = 5  # the number of parameters

    dsdp_expected = linspace(1.0, t*np, t*np).reshape((t, np))

    dxdp_expected = dsdp_expected[:n, :]
    dydp_expected = dsdp_expected[n:, :]

    # Set rows in dxdp corresponding to unstable variables to zero
    dxdp_expected[iunstable, :] = 0.0

    drdp_expected = M @ dsdp_expected

    # Get the H block in M (which is possibly different than original H, because of fixed variables!)
    dgdp = -drdp_expected[:n, :]
    dbdp =  drdp_expected[n:n + ml, :]
    dhdp = -drdp_expected[n + ml:, :]

    # Set the rows of dgdp corresponding to unstable variables
    dgdp[iunstable, :] = linspace(1.0, nu*np, nu*np).reshape((nu, np))

    # The solution of the sensitivity derivatives calculation
    dxdp = zeros((n, np))
    dydp = zeros((m, np))
    dzdp = zeros((n, np))

    stepper.sensitivities(dgdp, dhdp, dbdp, stability, dxdp, dydp, dzdp)

    # Assemble dsdp = [dxdp; dydp] and compute drdp = M*dsdp
    dsdp = block([[dxdp], [dydp]])
    drdp = M @ dsdp

    # Calculate the expected sensitivity derivatives dzdp = dgdp + tr(W)dydp
    dzdp_expected = dgdp + W.T @ dydp  # do not use dydp_expected here; causes failure when there are basic fixed variables (degeneracy)!!!
    dzdp_expected[istable, :] = 0.0  # dzdp is zero for stable variables

    # Print state variables
    def print_state():
        set_printoptions(suppress=True, linewidth=1000, precision=3)
        print()
        print(f"assemble_W  = {assemble_W}")
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
        print(f"dxdp(actual) = \n{dxdp}")
        print(f"dxdp(expected) = \n{dxdp_expected}")
        print(f"dydp(actual) = \n{dydp}")
        print(f"dydp(expected) = \n{dydp_expected}")
        print(f"dzdp(actual) = \n{dzdp}")
        print(f"dzdp(expected) = \n{dzdp_expected}")
        print()

    # Compare the actual and expected Newton steps
    s_actual = concatenate([dx, dy])

    assert allclose(M @ s_actual, M @ s_expected), print_state()

    # Compare the actual and expected right-hand side vector r of residuals
    r_actual = concatenate([rx, ry])

    assert allclose(r_actual, r_expected)

    # Compare the actual and expected z vector
    z_actual = z
    z_expected = g + W.T @ y

    assert allclose(z_actual, z_expected)

    # Create a Stability object with expected state
    expected_stability = create_expected_stability(A, x, b, z, xlower, xupper)

    check_stability(stability, expected_stability)

    # Compare the actual and expected sensitivity derivatives
    assert allclose(drdp_expected, drdp)
    assert allclose(dzdp_expected, dzdp)
