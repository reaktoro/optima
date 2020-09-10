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

from utils.matrices import assemble_matrix_Ax, matrix_non_singular, pascal_matrix

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
    [0, 1, 2]
]

# Tested cases for the indices of variables with lower bounds
tested_ilower = [
    [],
    [3, 4, 5]
]

# Tested cases for the indices of variables with upper bounds
tested_iupper = [
    [],
    [6, 7, 8]
]

# Tested cases for the saddle point methods
tested_methods = [
    SaddlePointMethod.Fullspace,
    SaddlePointMethod.Nullspace,
    SaddlePointMethod.Rangespace,
]


def create_objective_fn(Hxx, Hxp, cx):
    def fn(x, p, res):
        res.f   = 0.5 * (x.T @ Hxx @ x) + x.T @ Hxp @ p + cx.T @ x
        res.fx  = Hxx @ x + Hxp @ p + cx
        res.fxx = Hxx
        res.fxp = Hxp
    return fn


def create_constraint_hfn():
    def fn(x, p, res):
        pass
    return fn


def create_constraint_vfn(Vpx, Vpp, cp):
    def fn(x, p, res):
        res.h  = Vpx @ x + Vpp @ p + cp
        res.hx = Vpx
        res.hp = Vpp
    return fn



@mark.parametrize("np"    , tested_np)
@mark.parametrize("ny"    , tested_ny)
@mark.parametrize("nz"    , tested_nz)
@mark.parametrize("nbu"   , tested_nbu)
@mark.parametrize("nl"    , tested_nl)
@mark.parametrize("ifixed", tested_ifixed)
@mark.parametrize("ilower", tested_ilower)
@mark.parametrize("iupper", tested_iupper)
@mark.parametrize("method", tested_methods)
def test_solver(np, ny, nz, nbu, nl, ifixed, ilower, iupper, method):

    # Due to a current limitation in the algorithm, if number of parameter
    # variables is non-zero and number of linearly dependent or number of basic
    # unstable variables is non-zero, skip the test.
    if np > 0 and nbu + nl > 0:
        return

    # Skip if there are no unstable/fixed variables, and the number of unstable
    # basic variables is non-zero
    if nbu > 0 and len(ifixed) == 0:
        return

    # Skip if there are more Lagrange multipliers than primal variables
    if nx < ny + nz:
        return

    # Add the indices of fixed variables to those that have lower and upper bounds
    # since fixed variables are those that have identical lower and upper bounds
    ilower = list(set(ilower + ifixed))
    iupper = list(set(iupper + ifixed))

    # The number variables with finite lower and upper bounds, and fixed variables
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
    xlower = full(nx, -inf)
    xupper = full(nx,  inf)

    # Create vectors for the lower and upper bounds of the p variables
    plower = full(np, -inf)
    pupper = full(np,  inf)

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

    # Create vector u = (x, p, y, z) with the expected solution of the optimization problem
    u = linspace(1.0, t, t)

    # Get references to the subvectors x, p, y, z in u
    x, p, y, z = split(u, [nx, nx + np, nx + np + ny])

    # Set lower/upper unstable variables in x to their respective lower/upper bounds
    x[iunstable_lower] = xlower[iunstable_lower]
    x[iunstable_upper] = xupper[iunstable_upper]

    # Create the expected vector s = g + tr(Ax)*y + tr(Jx)*z
    s = zeros(nx)
    s[iunstable_lower] =  123  # lower unstable variables have positive value for s
    s[iunstable_upper] = -123  # upper unstable variables have negative value for s

    # Create matrices Hxx, Hxp, Vpx, Vpp
    Hxx = matrix_non_singular(nx)
    Hxp = pascal_matrix(nx, np) * 3  # ensure distinct from Vpx for more realistic cases
    Vpx = pascal_matrix(np, nx) * 5  # ensure distinct from Hxp for more realistic cases
    Vpp = matrix_non_singular(np)

    # Ensure Hxx is diagonal in case Rangespace method is used
    if method == SaddlePointMethod.Rangespace:
        Hxx = abs(diag(diag(Hxx)))

    # Zero out rows and columns in Hxx, Hxp, Vpx corresponding to fixed variables.
    # This is needed for consistent computation of vector cx below.
    Hxx[ifixed, :] = 0.0
    Hxx[:, ifixed] = 0.0
    Hxp[ifixed, :] = 0.0
    Vpx[:, ifixed] = 0.0

    # Compute the expected gradient vector at the solution using s = gx + tr(Ax)*y + tr(Jx)*z
    gx = s - Ax.T @ y - Jx.T @ z

    # Compute vector cx in f(x, p) = 1/2 tr(x)*Hxx*x + tr(x)*Hxp*p + tr(cx)*x using gx = Hxx*x + tr(p)*tr(Hxp) + cx
    cx = gx - (Hxx @ x) - (Hxp @ p).T

    # Compute vector cp in v(x, p) = Vpx*x + Vpp*p + cp, and since v(x,p) = 0, thus cp = -(Vpx*x + Vpp*p)
    cp = -(Vpx @ x + Vpp @ p)

    # Create the objective function with assembled Hxx, Hxp, cx
    obj = create_objective_fn(Hxx, Hxp, cx)

    # Create the nonlinear equality constraint function h(x, p)
    h = create_constraint_hfn()

    # Create the nonlinear external constraint function v(x, p)
    v = create_constraint_vfn(Vpx, Vpp, cp)

    # Compute vector b in Ax*x + Ap*p = b
    b = Ax @ x + Ap @ p

    # Keep references to current x, p, y, z, s as they are the expected solution
    x_expected = x
    p_expected = p
    y_expected = y
    z_expected = z
    s_expected = s

    # Create vectors for the solution of the optimization problem
    x = zeros(nx)
    p = zeros(np)
    y = zeros(ny)
    z = zeros(nz)
    s = zeros(nx)

    # Create the stability state of the variables
    stability = Stability()

    # Create the options for the optimization calculation
    options = Options()
    # options.output.active = True
    options.kkt.method = method
    options.max_iterations = 10

    dims = Dims()
    dims.x  = nx
    dims.p  = np
    dims.be = ny
    dims.bg = 0
    dims.he = 0
    dims.hg = 0

    problem = Problem(dims)
    problem.Aex = Ax
    problem.Aep = Ap
    problem.be  = b
    problem.xlower = xlower
    problem.xupper = xupper
    problem.plower = plower
    problem.pupper = pupper
    problem.f = obj
    problem.he = h
    problem.v = v

    # Solve the optimization problem
    solver = Solver(problem)
    solver.setOptions(options)

    state = State(dims)

    res = solver.solve(state, problem)

    if not res.succeeded:

        # set_printoptions(linewidth=100000, formatter={'float': '{: 0.3f}'.format})
        set_printoptions(linewidth=100000, precision=6, suppress=True)
        print()
        # print(f"H = \n{H}\n")
        # print(f"A = \n{A}\n")
        print(f"x(actual)   = {state.x}")
        print(f"x(expected) = {x_expected}")
        print(f"x(diff) = {abs(state.x - x_expected)}")
        print(f"p(actual)   = {state.p}")
        print(f"p(expected) = {p_expected}")
        print(f"p(diff) = {abs(state.p - p_expected)}")
        print(f"y(actual)   = {state.y}")
        print(f"y(expected) = {y_expected}")
        print(f"y(diff) = {abs(state.y - y_expected)}")
        print(f"z(actual)   = {state.z}")
        print(f"z(expected) = {z_expected}")
        print(f"z(diff) = {abs(state.z - z_expected)}")
        print(f"s(expected) = {s_expected}")
        print(f"s(diff) = {abs(state.s - s_expected)}")

    assert res.succeeded
