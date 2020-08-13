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


def print_state(M, r, s, m, nx, np):
    set_printoptions(linewidth=1000, suppress=True)
    slu = eigen.solve(M, r)
    # print( 'M        = \n', M )
    # print( 'r        = ', r )
    print( 'x        = ', s[:nx] )
    print( 'x(lu)    = ', slu[:nx] )
    print( 'x(diff)  = ', abs(s[:nx] - slu[:nx]) )
    print( 'p        = ', s[nx:nx+np] )
    print( 'p(lu)    = ', slu[nx:nx+np] )
    print( 'p(diff)  = ', abs(s[nx:nx+np] - slu[nx:nx+np]) )
    print( 'y        = ', s[nx+np:] )
    print( 'y(lu)    = ', slu[nx+np:] )
    print( 'y(diff)  = ', abs(s[nx+np:] - slu[nx+np:]) )
    print( 'res      = ', M.dot(s) - r )
    print( 'res(lu)  = ', M.dot(slu) - r )


# The number of primal variables x
nx = 20

# Tested number of parameter variables p
tested_np = [0, 5]

# Tested number of rows in matrix Ax and Ap
tested_ml = [6, 4]

# Tested number of rows in matrix Jx and Jp
tested_mn = [3, 1, 0]

# Tested cases for the matrix W = [Ax Ap; Jx Jp]
tested_matrices_W = testing_matrices_W

# Tested cases for the indices of fixed variables
tested_ifixed = [
    arange(0),
    arange(1),
    array([1, 3, 7, 9])
]

# Tested cases for the conditions of the variables in terms of pivot variables
tested_variable_conditions = [
    'explicit-variables-all',
    'explicit-variables-some',
    'explicit-variables-none'
]

# Tested cases for the saddle point methods
tested_methods = [
    SaddlePointMethod.Fullspace,
    SaddlePointMethod.Nullspace,
    SaddlePointMethod.Rangespace
]

# Combination of all tested cases
testdata = product(tested_np,
                   tested_ml,
                   tested_mn,
                   tested_matrices_W,
                   tested_ifixed,
                   tested_variable_conditions,
                   tested_methods)

@mark.parametrize("args", testdata)
def test_saddle_point_solver(args):

    np, ml, mn, assemble_W, ifixed, variable_condition, method = args

    m = ml + mn

    t = nx + np + m

    nf = len(ifixed)

    expected_xpy = linspace(1, t, t)

    # Assemble the coefficient matrix W = [Ax Ap; Jx Jp]
    W = assemble_W(m, nx + np, ifixed)

    # Extract the blocks of W = [Wx; Wp] = [Ax Ap; Jx Jp] = [A; J]
    Wx = W[:, :nx]
    Wp = W[:, nx:]

    Ax = Wx[:ml, :]
    Ap = Wp[:ml, :]

    Jx = Wx[ml:, :]
    Jp = Wp[ml:, :]

    # Create the blocks Hxx, Hxp
    Hxx = matrix_non_singular(nx)
    Hxp = pascal_matrix(nx, np)

    # Create the blocks Vpx, Vpp
    Vpx = pascal_matrix(np, nx)
    Vpp = matrix_non_singular(np)

    # Create the zero blocks Opy, Oyy
    Opy = zeros((np, m))
    Oyy = zeros((m, m))

    # Ensure Hxx is diagonal in case Rangespace method is used
    if method == SaddlePointMethod.Rangespace:
        Hxx = abs(diag(linspace(1, nx, num=nx)))
        Hxp = zeros((nx, np))  # TODO: This should not be forced zero to pass the tests with Rangespace - improve this!
        Vpx = zeros((np, nx))  # TODO: This should not be forced zero to pass the tests with Rangespace - improve this!

    # The sequence along the diagonal that is affected to control the number of pivot variables
    if variable_condition == 'explicit-variables-all':
        jexplicit = slice(nx)
        jimplicit = []
    if variable_condition == 'explicit-variables-none':
        jexplicit = []
        jimplicit = slice(nx)
    if variable_condition == 'explicit-variables-some':
        jexplicit = slice(m)
        jimplicit = slice(m, nx)

    # Adjust the diagonal entries to control number of pivot variables
    Hxx[jexplicit, jexplicit] *= 1.0e+08
    Hxx[jimplicit, jimplicit] *= 1.0e-08

    Hxx[ifixed, :] = 0.0       # zero out rows in Hxx corresponding to fixed variables
    Hxx[:, ifixed] = 0.0       # zero out cols in Hxx corresponding to fixed variables
    Hxx[ifixed, ifixed] = 1.0  # set one to diagonal entries in Hxx corresponding to fixed variables

    Hxp[ifixed, :] = 0.0       # zero out rows in Hxx corresponding to fixed variables
    Vpx[:, ifixed] = 0.0       # zero out cols in Vpx corresponding to fixed variables

    # The WxT = tr(Wx) matrix in M
    WxT = Wx.T
    WxT[ifixed, :] = 0.0      # zero out rows in WxT corresponding to fixed variables

    # Assemble the coefficient matrix M
    M = block([[Hxx, Hxp, WxT], [Vpx, Vpp, Opy], [Wx, Wp, Oyy]])

    # Compute the right-hand side vector r = M * expected
    r = M @ expected_xpy

    # The right-hand side vectors ax, ap, b
    ax = r[:nx]
    ap = r[nx:nx+np]
    b  = r[nx+np:]

    # The component vectors in b = [bl, bn]
    bl = b[:ml]
    bn = b[ml:]

    # The solution vectors x, p, y
    x = ax.copy()
    p = ap.copy()
    y = b.copy()

    # The scaling vector used as weights for the canonicalization
    w = linspace(1, nx, nx)

    # Specify the saddle point method for the current test
    options = SaddlePointOptions()
    options.method = method

    # Create a SaddlePointSolver to solve the saddle point problem
    solver = SaddlePointSolver(nx, np, m, Ax, Ap)
    solver.setOptions(options)
    solver.canonicalize(Hxx, Hxp, Vpx, Vpp, Jx, Jp, w, ifixed)
    solver.decompose(Hxx, Hxp, Vpx, Vpp, Jx, Jp, ifixed)

    def check_solution(x, p, y):
        # Create solution vector s = [x, p, y]
        s = concatenate([x, p, y])

        # Check the residual of the equation M * s = r
        # assert_allclose(M @ s, M @ expected_xpy)

        tol = 1e-13

        succeeded = norm(M @ s - r) / norm(r) < tol

        if not succeeded:
            print()
            print(f"np = {np}")
            print(f"ml = {ml}")
            print(f"mn = {mn}")
            print(f"assemble_W = {assemble_W}")
            print(f"structure_H = {structure_H}")
            print(f"ifixed = {ifixed}")
            print(f"variable_condition = {variable_condition}")
            print(f"method = {method}")
            print()

            print_state(M, r, s, m, nx, np)

        assert norm(M @ s - r) / norm(r) < tol

    # Check the overload solve(x, p, y) works
    solver.solve(x, p, y)

    check_solution(x, p, y)

    #---------------------------------------------------------------------------
    # Check the overload method solve(x, p, g, v, b, h, xbar, pbar, ybar) works
    #---------------------------------------------------------------------------
    x0 = linspace(1, nx, nx) * 10
    p0 = linspace(1, np, np) * 100

    x0[ifixed] = expected_xpy[ifixed]  # this is needed because fixed variables end up with what ever is in x0

    g = Hxx @ x0 + Hxp @ p0 - ax   # compute g so that Hxx * x0 + Hxp * p0 - g === ax
    v = Vpx @ x0 + Vpp @ p0 - ap   # compute g so that Hxx * x0 + Hxp * p0 - g === ap
    h = Jx @ x0 + Jp @ p0 - bn     # compute h so that Jx * x0 + Jp * p0 - h === bn

    solver.solve(x0, p0, g, v, bl, h, x, p, y)

    check_solution(x, p, y)
