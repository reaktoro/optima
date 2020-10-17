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

from utils.matrices import assemble_matrix_Ax, matrix_non_singular, pascal_matrix


def print_state(M, r, s, nx, np, ny, nz):
    slu = eigen.solve(M, r)
    # print( 'M        = \n', M )
    # print( 'r        = ', r )
    sx, sp, sy, sz = split(s, [nx, nx+np, nx+np+ny])
    slux, slup, sluy, sluz = split(slu, [nx, nx+np, nx+np+ny])
    print()
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
    print()


# The number of primal variables x
tested_nx = [10, 15, 20]

# Tested number of parameter variables p
tested_np = [0, 5]

# Tested number of Lagrange multipliers y (i.e., number of rows in A = [Ax Ap])
tested_ny = [4, 8]

# Tested number of Lagrange multipliers z (i.e., number of rows in J = [Jx Jp])
tested_nz = [0, 5]

# Tested number of unstable/fixed basic variables
tested_nbu = [0, 1, 2]

# Tested number of linearly dependent rows in Ax
tested_nl = [0, 1, 2]

# Tested cases for the indices of unstable variables (i.e. fixed at given values)
tested_ju = [
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

@mark.parametrize("nx"                , tested_nx)
@mark.parametrize("np"                , tested_np)
@mark.parametrize("ny"                , tested_ny)
@mark.parametrize("nz"                , tested_nz)
@mark.parametrize("nbu"               , tested_nbu)
@mark.parametrize("nl"                , tested_nl)
@mark.parametrize("ju"                , tested_ju)
@mark.parametrize("variable_condition", tested_variable_conditions)
@mark.parametrize("method"            , tested_methods)
def test_saddle_point_solver(nx, np, ny, nz, nbu, nl, ju, variable_condition, method):

    set_printoptions(linewidth=10000, suppress=True, precision=6)

    # The indices of the stable variables in x
    js = list(set(range(nx)) - set(ju))

    # Due to a current limitation in the algorithm, if number of parameter
    # variables is non-zero and number of linearly dependent or number of basic
    # unstable variables is non-zero, skip the test.
    if np > 0 and nbu + nl > 0:
        return

    # Skip if there are no unstable/fixed variables, and the number of unstable
    # basic variables is non-zero
    if nbu > 0 and len(ju) == 0:
        return

    # Skip if there are more Lagrange multipliers than primal variables
    if nx < ny + nz:
        return

    t = nx + np + ny + nz

    # Assemble the coefficient matrices Ax and Ap
    Ax = assemble_matrix_Ax(ny, nx, nbu, nl, ju)
    Ap = linspace(1, ny*np, ny*np).reshape((ny, np))

    # Assemble the coefficient matrix J = [Jx Jp]
    J = linspace(0, 1, nz*(nx + np)).reshape((nz, nx + np))

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

    # The sequence along the diagonal that is affected to control the number of pivot variables
    if variable_condition == 'explicit-variables-all':
        jexplicit = list(range(nx))
        jimplicit = []
    if variable_condition == 'explicit-variables-none':
        jexplicit = []
        jimplicit = list(range(nx))
    if variable_condition == 'explicit-variables-some':
        jexplicit = list(range(ny))
        jimplicit = list(range(ny, nx))

    # Adjust the diagonal entries to control number of pivot variables
    Hxx[jexplicit, jexplicit] *= 1.0e+08
    Hxx[jimplicit, jimplicit] *= 1.0e-08

    # Create the zero blocks Opz, Opy, Ozz, Ozy, Oyz, Oyy
    Opz = zeros((np, nz))
    Opy = zeros((np, ny))
    Ozz = zeros((nz, nz))
    Ozy = zeros((nz, ny))
    Oyz = zeros((ny, nz))
    Oyy = zeros((ny, ny))

    # Assemble the coefficient matrix M
    M = block([
        [Hxx, Hxp, Jx.T, Ax.T],
        [Vpx, Vpp,  Opz,  Opy],
        [ Jx,  Jp,  Ozz,  Ozy],
        [ Ax,  Ap,  Oyz,  Oyy]]
    )

    M[ju, :]  = 0.0  # zero out rows in M corresponding to fixed/unstable variables
    M[:, ju]  = 0.0  # zero out cols in M corresponding to fixed/unstable variables
    M[ju, ju] = 1.0  # set one to diagonal entries in M corresponding to fixed/unstable variables

    # The scaling vector used as weights for the canonicalization
    w = linspace(1, nx, nx)

    # Specify the saddle point method for the current test
    options = SaddlePointOptions()
    options.method = method

    # Create a SaddlePointSolver to solve the saddle point problem
    solver = SaddlePointSolver(nx, np, ny, nz, Ax, Ap)
    solver.setOptions(options)
    solver.canonicalize(Hxx, Hxp, Vpx, Vpp, Jx, Jp, ju, w)
    solver.decompose()

    def check_solution():
        succeeded = norm(M @ s - r) / norm(r) < 1e-12

        if not succeeded:
            print_state(M, r, s, nx, np, ny, nz)

        assert norm(M @ s - r) / norm(r) < 1e-12

    # Views to the matrices Hxx, Hxp, Vpx
    Hss = Hxx[js,:][:,js]
    Hsu = Hxx[js,:][:,ju]
    Hus = Hxx[ju,:][:,js]
    Huu = Hxx[ju,:][:,ju]
    Hsp = Hxp[js,:]
    Hup = Hxp[ju,:]
    Vps = Vpx[:, js]
    Vpu = Vpx[:, ju]

    #---------------------------------------------------------------
    # Check the overload rhs(ax, ap, ay, az) works
    #---------------------------------------------------------------

    # The actual solution vector s = (x, p, z, y)
    s = zeros(t)

    x, p, z, y = split(s, [nx, nx + np, nx + np + nz])

    # The expected solution vector s_exp = (x0, p0, z0, y0)
    s_exp = linspace(1, t, t)

    s_exp[ju] *= 999.9  # ensure large values for fixed variables in case debugging is needed

    # Compute the right-hand side vector r = M * expected
    r = M @ s_exp

    # The right-hand side vectors ax, ap, ay, az
    ax, ap, az, ay = split(r, [nx, nx + np, nx + np + nz])

    solver.rhs(ax, ap, ay, az)
    solver.solve(x, p, y, z)

    check_solution()

    #---------------------------------------------------------------
    # Check the overload method rhs(g, x, p, y, z, v, h, b) works
    #---------------------------------------------------------------
    x0 = linspace(1, nx, nx) * 10
    p0 = linspace(1, np, np) * 20
    y0 = linspace(1, ny, ny) * 30
    z0 = linspace(1, nz, nz) * 40

    # For this rhs overload, solution x satisfies x[iunstable] = 0
    s_exp[ju] = 0.0

    # Update the right-hand side vector r = (ax, ap, az, ay)
    r = M @ s_exp

    # The right-hand side vectors ax, ap, ay, az
    ax, ap, az, ay = split(r, [nx, nx + np, nx + np + nz])

    axs = ax[js]
    axu = ax[ju]

    As = Ax[:, js]
    Au = Ax[:, ju]
    Js = Jx[:, js]

    g0 = -ax - Ax.T @ y0 - Jx.T @ z0  # compute g0 using the fact that ax = -(g0 + tr(Ax)*y0 + tr(Jx)*z0)
    v0 = -ap                          # compute v0 using the fact that ap = -v0
    h0 = -az                          # compute h0 using the fact that az = -h0
    b0 =  ay + Ax @ x0 + Ap @ p0      # compute b0 using the fact that ay = -(Ax * x0 + Ap * p0 - b0)

    solver.rhs(g0, x0, p0, y0, z0, v0, h0, b0)
    solver.solve(x, p, y, z)

    check_solution()

    #---------------------------------------------------------------
    # Check the overload method rhs(g, x, p, v, h, b) works
    #---------------------------------------------------------------

    # Note: This overload method has been deprecated. The code below is kept
    # here just in case it becomes needed again.

    # # For this rhs overload, solution x satisfies x[iunstable] = x0[iunstable]
    # s_exp[ju] = x0[ju]

    # # Update the right-hand side vector r = (ax, ap, az, ay)
    # r = M @ s_exp

    # # The right-hand side vectors ax, ap, ay, az
    # ax, ap, az, ay = split(r, [nx, nx + np, nx + np + nz])

    # g0 = zeros(nx)
    # g0[js] = Hss @ x0[js] + Hsp @ p0 - ax[js]   # compute stable entries in g0 using the fact that as = Hss*xs0 + Hsp*p0 - gs0
    # g0[ju] = float("nan")                       # set to nan to ensure the unstable entries in g0 is indeed ignored
    # v0 = Vps @ x0[js] + Vpp @ p0 - ap   # compute v0 using the fact that ap = Vpx*x0 + Vpp*p0 - v0
    # h0 = Js @ x0[js] + Jp @ p0 - az     # compute h0 using the fact that az = Jx*x0 + Jp*p0 - h0
    # b0 = ay + Au @ x0[ju]               # compute b0 using the fact that ay = b0

    # solver.rhs(g0, x0, p0, v0, h0, b0)
    # solver.solve(x, p, y, z)

    # check_solution()

    #---------------------------------------------------------------
    # Check method multiply(rx, rp, ry, rz, ax, ap, ay, az) works
    #---------------------------------------------------------------
    r = linspace(1, t, t)
    a = zeros(t)

    rx, rp, rz, ry = split(r, [nx, nx+np, nx+np+nz])
    ax, ap, az, ay = split(a, [nx, nx+np, nx+np+nz])

    solver.multiply(rx, rp, ry, rz, ax, ap, ay, az)

    assert norm(M @ r - a) / norm(M @ r) < 1e-12

    #---------------------------------------------------------------
    # Check method transposeMultiply(rx, rp, ry, rz, ax, ap, ay, az) works
    #---------------------------------------------------------------
    r = linspace(1, t, t)
    a = zeros(t)

    rx, rp, rz, ry = split(r, [nx, nx+np, nx+np+nz])
    ax, ap, az, ay = split(a, [nx, nx+np, nx+np+nz])

    solver.transposeMultiply(rx, rp, ry, rz, ax, ap, ay, az)

    assert norm(M.T @ r - a) / norm(M.T @ r) < 1e-12
