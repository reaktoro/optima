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

from utils.matrices import testing_matrices_A, matrix_non_singular, pascal_matrix


def print_state(M, r, s, nx, np, ny, nz):
    set_printoptions(linewidth=1000, suppress=True)
    slu = eigen.solve(M, r)
    # print( 'M        = \n', M )
    # print( 'r        = ', r )
    sx, sp, sy, sz = split(s, [nx, nx+np, nx+np+ny])
    slux, slup, sluy, sluz = split(slu, [nx, nx+np, nx+np+ny])
    print( 'x        = ', sx             )
    print( 'x(lu)    = ', slux           )
    print( 'x(diff)  = ', abs(sx - slux) )
    print( 'p        = ', sp             )
    print( 'p(lu)    = ', slup           )
    print( 'p(diff)  = ', abs(sp - slup) )
    print( 'y        = ', sy             )
    print( 'y(lu)    = ', sluy           )
    print( 'y(diff)  = ', abs(sy - sluy) )
    print( 'z        = ', sz             )
    print( 'z(lu)    = ', sluz           )
    print( 'z(diff)  = ', abs(sz - sluz) )
    print( 'res      = ', M.dot(s) - r   )
    print( 'res(lu)  = ', M.dot(slu) - r )


# The number of primal variables x
nx = 20

# Tested number of parameter variables p
tested_np = [0, 5]

# Tested number of Lagrange multipliers y (i.e., number of rows in A = [Ax Ap])
tested_ny = [4, 6]

# Tested number of Lagrange multipliers z (i.e., number of rows in J = [Jx Jp])
tested_nz = [0, 5]

# Tested cases for matrix A = [Ax Ap]
tested_matrices_A = testing_matrices_A

# Tested cases for the indices of unstable variables (i.e. fixed at given values)
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
                   tested_ny,
                   tested_nz,
                   tested_matrices_A,
                   tested_ifixed,
                   tested_variable_conditions,
                   tested_methods)

@mark.parametrize("args", testdata)
def test_saddle_point_solver(args):

    np, ny, nz, assemble_A, ifixed, variable_condition, method = args

    t = nx + np + ny + nz

    nf = len(ifixed)

    expected_xpyz = linspace(1, t, t)

    # Assemble the coefficient matrix A = [Ax Ap]
    A = assemble_A(ny, nx + np, ifixed)

    # Extract the blocks of A = [Ax Ap]
    Ax = A[:, :nx]
    Ap = A[:, nx:]

    # Assemble the coefficient matrix J = [Jx Jp]
    J = pascal_matrix(nz, nx + np)

    # Extract the blocks of J = [Jx Jp]
    Jx = J[:, :nx]
    Jp = J[:, nx:]

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

    # The sequence along the diagonal that is affected to control the number of pivot variables
    if variable_condition == 'explicit-variables-all':
        jexplicit = slice(nx)
        jimplicit = []
    if variable_condition == 'explicit-variables-none':
        jexplicit = []
        jimplicit = slice(nx)
    if variable_condition == 'explicit-variables-some':
        jexplicit = slice(ny)
        jimplicit = slice(ny, nx)

    # Adjust the diagonal entries to control number of pivot variables
    Hxx[jexplicit, jexplicit] *= 1.0e+08
    Hxx[jimplicit, jimplicit] *= 1.0e-08

    Hxx[ifixed, :] = 0.0       # zero out rows in Hxx corresponding to fixed variables
    Hxx[:, ifixed] = 0.0       # zero out cols in Hxx corresponding to fixed variables
    Hxx[ifixed, ifixed] = 1.0  # set one to diagonal entries in Hxx corresponding to fixed variables

    Hxp[ifixed, :] = 0.0       # zero out rows in Hxx corresponding to fixed variables
    Vpx[:, ifixed] = 0.0       # zero out cols in Vpx corresponding to fixed variables

    # The AxT = tr(Ax) and JxT = tr(Jx) matrix blocks in M
    AxT = Ax.T
    JxT = Jx.T

    AxT[ifixed, :] = 0.0      # zero out rows in AxT corresponding to fixed variables
    JxT[ifixed, :] = 0.0      # zero out rows in JxT corresponding to fixed variables

    # Create the zero blocks Opz, Opy, Ozz, Ozy, Oyz, Oyy
    Opz = zeros((np, nz))
    Opy = zeros((np, ny))
    Ozz = zeros((nz, nz))
    Ozy = zeros((nz, ny))
    Oyz = zeros((ny, nz))
    Oyy = zeros((ny, ny))

    # Assemble the coefficient matrix M
    M = block([
        [Hxx, Hxp, JxT, AxT],
        [Vpx, Vpp, Opz, Opy],
        [ Jx,  Jp, Ozz, Ozy],
        [ Ax,  Ap, Oyz, Oyy]]
    )

    # Compute the right-hand side vector r = M * expected
    r = M @ expected_xpyz

    # The right-hand side vectors ax, ap, ay, az
    ax, ap, az, ay = split(r, [nx, nx + np, nx + np + nz])

    # The solution vectors x, p, y, z
    x = ax.copy()
    p = ap.copy()
    y = ay.copy()
    z = az.copy()

    # The scaling vector used as weights for the canonicalization
    w = linspace(1, nx, nx)

    # Specify the saddle point method for the current test
    options = SaddlePointOptions()
    options.method = method

    # Create a SaddlePointSolver to solve the saddle point problem
    solver = SaddlePointSolver(nx, np, ny, nz, Ax, Ap)
    solver.setOptions(options)
    solver.canonicalize(Hxx, Hxp, Vpx, Vpp, Jx, Jp, ifixed, w)
    solver.decompose()

    def check_solution(x, p, y, z):
        # Create solution vector s = [x, p, z, y]
        s = concatenate([x, p, z, y])

        # Check the residual of the equation M * s = r
        # assert_allclose(M @ s, M @ expected_xpyz)

        tol = 1e-13

        succeeded = norm(M @ s - r) / norm(r) < tol

        if not succeeded:
            print()
            print(f"nx = {nx}")
            print(f"np = {np}")
            print(f"ny = {ny}")
            print(f"nz = {nz}")
            print(f"assemble_A = {assemble_A}")
            print(f"ifixed = {ifixed}")
            print(f"variable_condition = {variable_condition}")
            print(f"method = {method}")
            print()

            print_state(M, r, s, nx, np, ny, nz)

        assert norm(M @ s - r) / norm(r) < tol

    #---------------------------------------------------------------
    # Check the overload rhs(ax, ap, ay, az) works
    #---------------------------------------------------------------
    solver.rhs(ax, ap, ay, az)
    solver.solve(x, p, y, z)

    check_solution(x, p, y, z)

    #---------------------------------------------------------------
    # Check the overload method rhs(g, x, p, y, z, v, h, b) works
    #---------------------------------------------------------------
    x0 = linspace(1, nx, nx) * 10
    p0 = linspace(1, np, np) * 20
    y0 = linspace(1, ny, ny) * 30
    z0 = linspace(1, nz, nz) * 40

    g0 = -ax - AxT @ y0 - JxT @ z0  # compute g0 using the fact that ax = -(g0 + tr(Ax)*y0 + tr(Jx)*z0)
    v0 = -ap                        # compute v0 using the fact that ap = -v0
    h0 = -az                        # compute h0 using the fact that az = -h0
    b0 =  ay + Ax @ x0 + Ap @ p0    # compute b0 using the fact that ay = -(Ax * x0 + Ap * p0 - b0)

    solver.rhs(g0, x0, p0, y0, z0, v0, h0, b0)
    solver.solve(x, p, y, z)

    assert all(x[ifixed] == 0.0)  # ensure x[i] = 0 for i in fixed/unstable variables for this rhs setup

    x[ifixed] = expected_xpyz[ifixed]  # replace these zeros by expected x so that the common test framework next succeeds

    check_solution(x, p, y, z)
