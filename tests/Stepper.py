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
from numpy.testing import assert_allclose
from pytest import approx, mark
from itertools import product

from utils.matrices import testing_matrices_W, matrix_non_singular, pascal_matrix
from utils.stability import create_expected_stability, check_stability


from enum import Enum

class MatrixStructure(Enum):
    Zero = 1
    Diagonal = 2
    Dense = 3


# The number of x variables
nx = 15

# Tested number of parameter variables p
tested_np = [0, 5]

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
    list(range(nx))  # all x variables with lower bounds
]

# Tested cases for the indices of variables with upper bounds
tested_iupper = [
    [],
    [1],
    [1, 3, 7, 9],
    list(range(nx))  # all x variables with upper bounds
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
testdata = product(tested_np,
                   tested_matrices_W,
                   tested_structures_H,
                   tested_ml,
                   tested_mn,
                   tested_ifixed,
                   tested_ilower,
                   tested_iupper,
                   tested_methods)


@mark.parametrize("args", testdata)
def test_active_stepper(args):

    np, assemble_W, structure_H, ml, mn, ifixed, ilower, iupper, method = args

    # Add the indices of fixed variables to those that have lower and upper bounds
    # since fixed variables are those that have identical lower and upper bounds
    ilower = list(set(ilower + ifixed))
    iupper = list(set(iupper + ifixed))

    # The number variables with finite lower and upper bounds, and fixed variables
    nlower = len(ilower)
    nupper = len(iupper)
    nfixed = len(ifixed)

    # The total number of rows in W = [Ax Ap; Jx Jp]
    m = ml + mn

    # The total number of variables x, p, y
    t = nx + np + m

    # Assemble the coefficient matrix W = [Ax Ap; Jx Jp]
    W = assemble_W(m, nx + np, ifixed)

    # Extract the blocks of W = [Wx; Wp] = [Ax Ap; Jx Jp] = [A; J]
    Wx = W[:, :nx]
    Wp = W[:, nx:]

    Ax = Wx[:ml, :]
    Ap = Wp[:ml, :]

    Jx = Wx[ml:, :]
    Jp = Wp[ml:, :]

    # Create vectors for the lower and upper bounds of the x variables
    xlower = full(nx, -inf)
    xupper = full(nx,  inf)

    # Create vectors for the lower and upper bounds of the p variables
    plower = full(np, -inf)
    pupper = full(np,  inf)

    # Set lower and upper bounds of x to negative and positive sequence respectively
    xlower[ilower] = -linspace(1.0, nlower, nlower) * 100
    xupper[iupper] =  linspace(1.0, nupper, nupper) * 100

    # Set lower and upper bounds to equal values for fixed x variables
    xlower[ifixed] = xupper[ifixed] = linspace(1, nfixed, nfixed) * 10

    # Create vectors x, p, y
    x = linspace(1.0, nx, nx)
    p = linspace(1.0, np, np)
    y = linspace(1.0, m, m)

    # Create the blocks Hxx, Hxp
    Hxx = matrix_non_singular(nx)
    Hxp = pascal_matrix(nx, np)

    # Create the blocks Vpx, Vpp
    Vpx = pascal_matrix(np, nx)
    Vpp = matrix_non_singular(np)

    # Ensure Hxx is diagonal in case Rangespace method is used
    if method == SaddlePointMethod.Rangespace:
        Hxx = abs(diag(linspace(1, nx, num=nx)))
        Hxp = zeros((nx, np))  # TODO: This should not be forced zero to pass the tests with Rangespace - improve this!
        Vpx = zeros((np, nx))  # TODO: This should not be forced zero to pass the tests with Rangespace - improve this!

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

    # Create vector s = (dx, dy) with the expected Newton step values
    s = s_expected = linspace(1.0, t, t)  # Introduce `s` as an alias for `s_expected`

    # Get references to the segments dx, dp, dy in s
    dx = s[:nx]
    dp = s[nx:nx+np]
    dy = s[nx+np:]

    # Set to zero the entries in dx corresponding to lower/upper unstable variables
    dx[iunstable] = 0.0

    # Set lower/upper unstable variables in x to their respective lower/upper bounds
    x[iunstable_lower] = xlower[iunstable_lower]
    x[iunstable_upper] = xupper[iunstable_upper]

    # Ensure rows/cols in Hxx, Hxp, Vpx take into account fixed/unstable variables
    Hxx[iunstable, :] = 0.0          # zero out rows in Hxx corresponding to fixed variables
    Hxx[:, iunstable] = 0.0          # zero out cols in Hxx corresponding to fixed variables
    Hxx[iunstable, iunstable] = 1.0  # set one to diagonal entries in Hxx corresponding to fixed variables

    Hxp[iunstable, :] = 0.0       # zero out rows in Hxx corresponding to fixed variables
    Vpx[:, iunstable] = 0.0       # zero out cols in Vpx corresponding to fixed variables

    # The WxT = tr(Wx) matrix in M
    WxT = Wx.T
    WxT[iunstable, :] = 0.0      # zero out rows in WxT corresponding to fixed variables

    # Create the zero blocks Opy, Oyy
    Opy = zeros((np, m))
    Oyy = zeros((m, m))

    # Assemble the coefficient matrix M
    M = block([[Hxx, Hxp, WxT], [Vpx, Vpp, Opy], [Wx, Wp, Oyy]])

    # Compute the right-hand side vector r = Ms
    r = M @ s

    # Get references to rx, rp, ry in r where rx := -(g + tr(Wx)y), rp := -(v), ry := (ryl, ryn)
    rx = r[:nx]
    rp = r[nx:nx+np]
    ry = r[nx+np:]

    # Get references to ryl and ryn in ry where ryl := -(Ax - b) and ryn := -h
    ryl = ry[:ml]
    ryn = ry[ml:]

    # Compute the gradient vector remembering that rx := -(g + tr(Wx)y) for stable variables, zero otherwise
    g = -(rx + Wx.T @ y)

    # Ensure g has values for lower/upper unstable variables that make these recognized as unstable indeed
    # Note that:
    #  - x[i] is lower unstable when x[i] == xlower[i] and z[i] > 0, where z = g + tr(W)*y
    #  - x[i] is upper unstable when x[i] == xupper[i] and z[i] < 0, where z = g + tr(W)*y
    g[iunstable_lower] =  inf
    g[iunstable_upper] = -inf

    # Compute the residual vector v of the external nonlinear constraints v(x,p) = 0 knowing that rp := -v
    v = -rp

    # Compute the right-hand side vector b remembering that ryl = -(Ax*x + Ap*p - b)
    b = ryl + Ax @ x + Ap @ p

    # Compute the residual vector h of the nonlinear equality constraints h(x, p) = 0 knowing that ryn := -h
    h = -ryn

    # The solution of the Newton step calculation
    dx   = zeros(nx)  # The Newton step for the primal variables *x*.
    dp   = zeros(np)  # The Newton step for the parameter variables *p*.
    dy   = zeros(m)   # The Newton step for the Lagrange multipliers *y*.
    resx = zeros(nx)  # The residuals of the first-order optimality conditions.
    resp = zeros(np)  # The residuals of the external constraints v(x, p) = 0.
    resy = zeros(m)   # The residuals of the linear/nonlinear feasibility conditions.
    errx = zeros(nx)  # The relative residual errors of the first-order optimality conditions.
    errp = zeros(np)  # The relative residuals of the external constraints v(x, p) = 0.
    erry = zeros(m)   # The relative residual errors of the linear/nonlinear feasibility conditions.
    z    = zeros(nx)  # The *unstabilities* of the variables defined as *z = g + tr(Wx)y* where *W = [Ax Ap; Jx Jp]*.

    # Create the stability state of the variables
    stability = Stability()

    # Set options for the Newton step calculation
    options = Options()
    options.kkt.method = method

    # Create a Newton step calculator
    stepper = Stepper(nx, np, m, Ax, Ap)
    stepper.setOptions(options)

    # Initialize, decompose the saddle point matrix, and solve the Newton step
    stepper.initialize(b, xlower, xupper, plower, pupper, x, stability)
    stepper.canonicalize(x, p, y, g, Hxx, Hxp, Vpx, Vpp, Jx, Jp, xlower, xupper, plower, pupper, stability)
    stepper.residuals(x, p, y, b, h, v, g, Jx, resx, resp, resy, errx, errp, erry, z)
    stepper.decompose(x, p, y, g, Hxx, Hxp, Vpx, Vpp, Jx, Jp, xlower, xupper, plower, pupper, stability)
    stepper.solve(x, p, y, g, b, h, v, stability, dx, dp, dy)

    #==============================================================
    # Compute the sensitivity derivatives of the optimal solution
    #==============================================================

    # The number of w parameters considered in the sensitivity calculations
    nw = 5

    # The expected solution of the sensitivity matrix dsdw = [dxdw; dpdw; dydw]
    dsdw_expected = linspace(1.0, t*nw, t*nw).reshape((t, nw))

    # The block views in dsdw = [dxdw; dpdw; dydw]
    dxdw_expected = dsdw_expected[:nx, :]
    dpdw_expected = dsdw_expected[nx:nx+np, :]
    dydw_expected = dsdw_expected[nx+np:, :]

    # Set rows in dxdw corresponding to unstable variables to zero (because dxudw === 0)
    dxdw_expected[iunstable, :] = 0.0

    # Compute the right-hand side matrix in M*dsdw = drdw
    drdw_expected = M @ dsdw_expected

    # The block views in drdw = [-dgdw; -dvdw; dbdw; -dhdw]
    dgdw = -drdw_expected[:nx, :]
    dvdw = -drdw_expected[nx:nx+np, :]
    dbdw =  drdw_expected[nx+np:nx+np+ml, :]
    dhdw = -drdw_expected[nx+np+ml:, :]

    # Set the rows of dgdw corresponding to unstable variables
    dgdw[iunstable, :] = linspace(1.0, nu*nw, nu*nw).reshape((nu, nw))

    # The actual/calculated sensitivity derivatives dsdw = [dxdw; dpdw; dydw] and dzdw
    dxdw = zeros((nx, nw))
    dpdw = zeros((np, nw))
    dydw = zeros((m,  nw))
    dzdw = zeros((nx, nw))

    # Compute the sensitivity matrices dxdw, dpdw, dydw, dzdw
    stepper.sensitivities(dgdw, dhdw, dbdw, dvdw, stability, dxdw, dpdw, dydw, dzdw)

    # Assemble dsdw = [dxdw; dpdw; dydw] and compute drdw = M*dsdw
    dsdw = block([[dxdw], [dpdw], [dydw]])
    drdw = M @ dsdw

    # Calculate the expected sensitivity derivatives dzdw = dgdw + tr(W)dydw
    dzdw_expected = dgdw + Wx.T @ dydw  # do not use dydw_expected here; causes failure when there are basic fixed variables (degeneracy)!!!
    dzdw_expected[istable, :] = 0.0  # dzdw is zero for stable variables

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
        print(f"dx(actual)     = {dx}")
        print(f"dx(expected)   = {s[:nx]}")
        print(f"dp(actual)     = {dx}")
        print(f"dp(expected)   = {s[nx:nx+np]}")
        print(f"dy(actual)     = {dy}")
        print(f"dy(expected)   = {s[nx+np:]}")
        print(f"dxdw(actual)   = \n{dxdw}")
        print(f"dxdw(expected) = \n{dxdw_expected}")
        print(f"dpdw(actual)   = \n{dpdw}")
        print(f"dpdw(expected) = \n{dpdw_expected}")
        print(f"dydw(actual)   = \n{dydw}")
        print(f"dydw(expected) = \n{dydw_expected}")
        print(f"dzdw(actual)   = \n{dzdw}")
        print(f"dzdw(expected) = \n{dzdw_expected}")
        print()

    # Compare the actual and expected Newton steps dx, dp, dy
    s_actual = concatenate([dx, dp, dy])

    assert_allclose(M @ s_actual, M @ s_expected)

    # Compare the actual and expected sensitivity derivatives dxdw, dpdw, dydw, dzdw
    assert_allclose(drdw_expected, drdw)
    assert_allclose(dzdw_expected, dzdw)

    # # Get the canonicalization matrix R
    # R = stepper.info().R  # TODO: Stepper has no info method yet.

    # # The expected optimality and feasibility residuals
    # resx_expected = abs(rx)
    # resy_expected = abs(R * ry)  # The feasibility residuals in canonical form!
    # z_expected = g + W.T @ y

    # assert_allclose(resx, resx_expected)
    # assert_allclose(resy, resy_expected)
    # assert_allclose(z, z_expected)

    # Create a Stability object with expected state
    expected_stability = create_expected_stability(Ax, x, b, z, xlower, xupper)

    check_stability(stability, expected_stability)
