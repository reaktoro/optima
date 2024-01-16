# Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
#
# Copyright © 2020-2024 Allan Leal
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


from testing.optima import *
from testing.utils.matrices import *


tested_nx      = [15, 20, 30, 50]  # The tested number of x variables
tested_np      = [0, 5]            # The tested number of p variables
tested_ny      = [5, 8]            # The tested number of y variables
tested_nz      = [0, 5]            # The tested number of z variables
tested_nl      = [0, 2]            # The tested number of linearly dependent rows in Ax
tested_nu      = [0, 2]            # The tested number of unstable variables
tested_diagHxx = [False, True]     # The tested options for Hxx structure

# Tested cases for the linear solver methods
tested_methods = [
    LinearSolverMethod.Fullspace,
    LinearSolverMethod.Nullspace,
    LinearSolverMethod.Rangespace
]

@pytest.mark.parametrize("nx"     , tested_nx)
@pytest.mark.parametrize("np"     , tested_np)
@pytest.mark.parametrize("ny"     , tested_ny)
@pytest.mark.parametrize("nz"     , tested_nz)
@pytest.mark.parametrize("nl"     , tested_nl)
@pytest.mark.parametrize("nu"     , tested_nu)
@pytest.mark.parametrize("diagHxx", tested_diagHxx)
@pytest.mark.parametrize("method" , tested_methods)
def testLinearSolver(nx, np, ny, nz, nl, nu, diagHxx, method):

    params = MasterParams(nx, np, ny, nz, nl, nu, diagHxx)

    if params.invalid(): return

    dims = params.dims

    if method == LinearSolverMethod.Rangespace and not diagHxx:
        return  # Rangespace method only applicable to diagonal Hxx matrices - consider here only when both are true

    M = createMasterMatrix(params)

    nw = params.dims.nw

    uexp = MasterVector(dims)
    uexp.x = npy.linspace(1, nx, nx)
    uexp.p = npy.linspace(1, np, np)
    uexp.w = npy.linspace(1, nw, nw)

    a = M * uexp

    canonicalizer = Canonicalizer(M)

    Mc = canonicalizer.canonicalMatrix()

    options = LinearSolverOptions()
    options.method = method

    linearsolver = LinearSolver()
    linearsolver.setOptions(options)

    u = MasterVector(dims)

    linearsolver.decompose(Mc)
    linearsolver.solve(Mc, a, u)

    assert_almost_equal( (M * u).array(), a.array() )

    ju = M.ju  # the indices of the unstable variables in x

    assert all(u.x[ju] == a.x[ju])  # ensure ux[ju] == ax[ju]
