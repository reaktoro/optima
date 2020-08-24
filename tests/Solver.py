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

from utils.matrices import testing_matrices_W, matrix_non_singular, pascal_matrix

# The number of x variables
nx = 15

# The number of equality constraints
m = 5

# The tested number of p variables
tested_np = [0, 5]

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
testdata = product(tested_np,
                   tested_matrices_W,
                   tested_structures_H,
                   tested_ifixed,
                   tested_ilower,
                   tested_iupper,
                   tested_methods)


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


@mark.parametrize("args", testdata)
def test_basic_solver(args):

    np, assemble_W, structure_H, ifixed, ilower, iupper, method = args

    # Add the indices of fixed variables to those that have lower and upper bounds
    # since fixed variables are those that have identical lower and upper bounds
    ilower = list(set(ilower + ifixed))
    iupper = list(set(iupper + ifixed))

    # The number variables with finite lower and upper bounds, and fixed variables
    nlower = len(ilower)
    nupper = len(iupper)
    nfixed = len(ifixed)

    # The total number of variables x, p, y
    t = nx + np + m

    # Assemble the coefficient matrix W = [Ax Ap; Jx Jp]
    W = assemble_W(m, nx + np, ifixed)

    # For the moment, set h = 0
    ml = m
    mn = 0

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

    # Get references to the subvectors x, p, y in s
    x = s[:nx]
    p = s[nx:nx+np]
    y = s[nx+np:]

    # Set lower/upper unstable variables in x to their respective lower/upper bounds
    x[iunstable_lower] = xlower[iunstable_lower]
    x[iunstable_upper] = xupper[iunstable_upper]

    # Create the expected vector z = g + tr(A)yl + tr(J)yn
    z = zeros(nx)
    z[iunstable_lower] =  123  # lower unstable variables have positive value for z
    z[iunstable_upper] = -123  # upper unstable variables have negative value for z

    # Create matrices Hxx, Hxp, Vpx, Vpp
    Hxx = matrix_non_singular(nx)
    Hxp = pascal_matrix(nx, np) * 3  # ensure distinct from Vpx for more realistic cases
    Vpx = pascal_matrix(np, nx) * 5  # ensure distinct from Hxp for more realistic cases
    Vpp = matrix_non_singular(np)

    # Ensure Hxx is diagonal in case Rangespace method is used
    if method == SaddlePointMethod.Rangespace:
        Hxx = abs(diag(diag(Hxx)))
        Hxp = zeros((nx, np))  # TODO: This should not be forced zero to pass the tests with Rangespace - improve this!
        Vpx = zeros((np, nx))  # TODO: This should not be forced zero to pass the tests with Rangespace - improve this!

    # Zero out rows and columns in Hxx, Hxp, Vpx corresponding to fixed variables.
    # This is needed for consistent computation of vector cx below.
    Hxx[ifixed, :] = 0.0
    Hxx[:, ifixed] = 0.0
    Hxp[ifixed, :] = 0.0
    Vpx[:, ifixed] = 0.0

    # Compute the expected gradient vector at the solution using z = gx + tr(Ax)*yl + tr(Jx)*yn = gx + tr(Wx)*y
    gx = z - Wx.T @ y

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

    # Keep references to current x, p, y, z as they are the expected solution
    x_expected = x
    p_expected = p
    y_expected = y
    z_expected = z

    # Create vectors for the solution of the optimization problem
    x = zeros(nx)
    p = zeros(np)
    y = zeros(m)
    z = zeros(nx)

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
    dims.be = m
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

    assert res.succeeded
