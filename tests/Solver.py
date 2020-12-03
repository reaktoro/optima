# Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
#
# Copyright (C) 2020 Allan Leal
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


from testing.optima import *
from testing.utils.matrices import *
from numpy import *



tested_nx      = [10, 15, 20, 30] # The tested number of x variables
tested_np      = [0, 5, 10, 15]   # The tested number of p variables
tested_ny      = [5, 10]          # The tested number of y variables
tested_nz      = [0, 5]           # The tested number of z variables
tested_nl      = [0, 2]           # The tested number of linearly dependent rows in Ax
tested_nul     = [0, 1, 5]        # The tested number of lower unstable variables
tested_nuu     = [0, 1, 5]        # The tested number of upper unstable variables
tested_diagHxx = [False, True]    # The tested diagonal structure of Hxx matrix

@pytest.mark.parametrize("nx"     , tested_nx)
@pytest.mark.parametrize("np"     , tested_np)
@pytest.mark.parametrize("ny"     , tested_ny)
@pytest.mark.parametrize("nz"     , tested_nz)
@pytest.mark.parametrize("nl"     , tested_nl)
@pytest.mark.parametrize("nul"    , tested_nul)
@pytest.mark.parametrize("nuu"    , tested_nuu)
@pytest.mark.parametrize("diagHxx", tested_diagHxx)
def testMasterSolver(nx, np, ny, nz, nl, nul, nuu, diagHxx):

    nw = ny + nz

    if nx <= nw: return
    if nx <= nul + nuu + nw: return
    if ny <= nl: return

    jul = range(nul)            # the indices of the expected lower unstable variables
    juu = range(nul, nul + nuu)  # the indices of the expected upper unstable variables

    Hxx = random.rand(nx, nx)
    Hxp = random.rand(nx, np)
    Vpx = random.rand(np, nx)
    Vpp = random.rand(np, np)
    Ax  = random.rand(ny, nx)
    Ap  = random.rand(ny, np)
    Jx  = random.rand(nz, nx)
    Jp  = random.rand(nz, np)

    Ax[ny-nl:, :] = 0.0  # last nl rows in Ax are forced to be linearly dependent
    Hxx = Hxx.T @ Hxx    # this ensures Hxx is positive semi-definite or definite

    if diagHxx:
        Hxx = diag(random.rand(nx))

    cx = ones(nx)
    cp = ones(np)
    cz = ones(nz)

    Hxx[jul, jul] = 1e6  # this ensures variables expected on their lower bounds are marked as unstable
    Hxx[juu, juu] = 1e6  # this ensures variables expected on their upper bounds are marked as unstable

    #-------------------------------------------------------------------------------
    # Uncomment the following code to enable slack variables that will ensure
    # feasible solutions. If the problem is feasible considering only the main
    # variables, these slack variables will become zero at the end of the calculation.
    #-------------------------------------------------------------------------------
    # Hxx[-nw:, :] = 0.0
    # Hxx[:, -nw:] = 0.0
    # Hxx[-nw:, -nw:] = eye(nw)
    # Hxp[-nw:, :] = 0.0

    # Ax[:, -nw:] = 0.0
    # Ax[:, -nw:-nw+ny] = eye(ny)

    # Jx[:, -nw:] = 0.0
    # Jx[:, -nz:] = eye(nz)

    # cx[-nw:] = 0.0
    #-------------------------------------------------------------------------------


    # print(f"Hxx = \n{Hxx}")

    def objectivefn_f(res, x, p, opts):
        dx = x - cx
        dp = p - cp
        res.f   = 0.5 * dx.T @ Hxx @ dx + dx.T @ Hxp @ dp
        res.fx  = Hxx @ dx + Hxp @ dp
        res.fxx = Hxx
        res.fxp = Hxp
        res.diagfxx = diagHxx
        res.fxx4basicvars = False
        res.succeeded = True

    def constraintfn_h(res, x, p, opts):
        res.val = Jx @ (x - cx) + Jp @ (p - cp)
        res.ddx = Jx
        res.ddp = Jp
        res.ddx4basicvars = False
        res.succeeded = True

    def constraintfn_v(res, x, p, opts):
        res.val = Vpx @ (x - cx) + Vpp @ (p - cp)
        res.ddx = Vpx
        res.ddp = Vpp
        res.ddx4basicvars = False
        res.succeeded = True

    xlower = full(nx, -inf)
    xupper = full(nx,  inf)

    xlower[jul] = 1.5  # this should be greater than 1.0
    xupper[jul] = 2.0  # this should be greater than xlower[jul]

    xlower[juu] = 0.0  # this should be less than xupper[juu]
    xupper[juu] = 0.5  # this should be less than 1.0

    dims = Dims()
    dims.x = nx
    dims.p = np
    dims.be = ny
    dims.he = nz

    problem = Problem(dims)
    problem.f = objectivefn_f
    problem.he = constraintfn_h
    problem.v = constraintfn_v
    problem.Aex = Ax
    problem.Aep = Ap
    problem.be = Ax @ cx + Ap @ cp
    problem.xlower = xlower
    problem.xupper = xupper
    problem.plower = full(np, -inf)
    problem.pupper = full(np,  inf)

    options = Options()
    # options.output.active = True

    options.newtonstep.linearsolver.method = \
        LinearSolverMethod.Rangespace if diagHxx else \
        LinearSolverMethod.Nullspace

    solver = Solver(problem)
    solver.setOptions(options)

    state = State(dims)

    res = solver.solve(problem, state)

    assert res.succeeded
