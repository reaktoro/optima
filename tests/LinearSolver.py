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
from numpy.testing import assert_almost_equal
from pytest import approx, mark
from utils.matrices import createMasterMatrix

tested_nx  = [15, 20]  # The tested number of x variables
tested_np  = [0, 5]    # The tested number of p variables
tested_ny  = [5, 10]   # The tested number of y variables
tested_nz  = [0, 5]    # The tested number of z variables
tested_nl  = [0, 2]    # The tested number of linearly dependent rows in Ax
tested_nu  = [0, 2]    # The tested number of unstable variables

# tested_nx  = [10]  # The tested number of x variables
# tested_np  = [2]    # The tested number of p variables
# tested_ny  = [5]   # The tested number of y variables
# tested_nz  = [2]    # The tested number of z variables
# tested_nl  = [2]    # The tested number of linearly dependent rows in Ax
# tested_nu  = [0]    # The tested number of unstable variables

# Tested cases for the linear solver methods
tested_methods = [
    LinearSolverMethod.Fullspace,
    LinearSolverMethod.Nullspace,
    LinearSolverMethod.Rangespace
]

@mark.parametrize("nx"    , tested_nx)
@mark.parametrize("np"    , tested_np)
@mark.parametrize("ny"    , tested_ny)
@mark.parametrize("nz"    , tested_nz)
@mark.parametrize("nl"    , tested_nl)
@mark.parametrize("nu"    , tested_nu)
@mark.parametrize("method", tested_methods)
def test_linear_solver(nx, np, ny, nz, nl, nu, method):

    nw = ny + nz

    # Ensure nx is larger than np and nw
    if nx < np or nx < nw: return

    # Ensure nl < ny
    if ny <= nl: return

    # Ensure nl < ny
    if ny <= nl: return

    diagHxx = True if method == LinearSolverMethod.Rangespace else False

    M = createMasterMatrix(nx, np, ny, nz, nl, nu, diagHxx)

    t = nx + np + ny + nz

    uexp = MasterVector(nx, np, nw)
    uexp.x = linspace(1, nx, nx)
    uexp.p = linspace(1, np, np)
    uexp.w = linspace(1, nw, nw)

    a = M * uexp

    Mc = CanonicalMatrix(nx, np, ny, nz)
    Mc.update(M)

    Mbar = Mc.view()

    dims = Mbar.dims
    nbs = dims.nbs

    options = LinearSolverOptions()
    options.method = method

    linearsolver = LinearSolver(nx, np, ny, nz)
    linearsolver.setOptions(options)

    u = MasterVector(nx, np, nw)

    linearsolver.decompose(Mbar)
    linearsolver.solve(Mbar, a, u)

    assert_almost_equal( (M * u).array(), a.array() )
