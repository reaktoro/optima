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
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


from testing.optima import *
from testing.utils.matrices import *


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
def testSolver(nx, np, ny, nz, nl, nul, nuu, diagHxx):

    nw = ny + nz

    if nx <= nw: return
    if nx <= nul + nuu + nw: return
    if ny <= nl: return

    jul = range(nul)             # the indices of the expected lower unstable variables
    juu = range(nul, nul + nuu)  # the indices of the expected upper unstable variables
    ju = list(jul) + list(juu)

    Hxx = rng.rand(nx, nx)
    Hxp = rng.rand(nx, np)
    Vpx = rng.rand(np, nx)
    Vpp = rng.rand(np, np)
    Ax  = rng.rand(ny, nx)
    Ap  = rng.rand(ny, np)
    Jx  = rng.rand(nz, nx)
    Jp  = rng.rand(nz, np)

    Ax[ny - nl:, :] = 0.0  # set last nl rows to be zero so that we have nl linearly dependent rows in Ax
    Ap[ny - nl:, :] = 0.0  # do the same to Ap, otherwise, expected error: Your matrix Ax is rank-deficient and matrix Ap is non-zero such that...

    Hxx = Hxx.T @ Hxx    # this ensures Hxx is positive semi-definite or definite

    if diagHxx:
        Hxx = npy.diag(rng.rand(nx))

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

    Ixx = npy.eye(nx)
    Oxp = npy.zeros((nx, np))
    Oxy = npy.zeros((nx, ny))
    Oxz = npy.zeros((nx, nz))

    Opx = npy.zeros((np, nx))
    Ipp = npy.eye(np)
    Opy = npy.zeros((np, ny))
    Opz = npy.zeros((np, nz))

    Oyx = npy.zeros((ny, nx))
    Oyp = npy.zeros((ny, np))
    Iyy = npy.eye(ny)
    Oyz = npy.zeros((ny, nz))

    Ozx = npy.zeros((nz, nx))
    Ozp = npy.zeros((nz, np))
    Ozy = npy.zeros((nz, ny))
    Izz = npy.eye(nz)

    nc = nx + np + ny + nz

    cx = rng.rand(nx)  # c = [cx, cp, cy, cz] -- the parameters used for sensitivity derivative calculations
    cp = rng.rand(np)
    cy = Ax @ cx + Ap @ cp
    cz = rng.rand(nz)

    cx[ju] = +1.0e4  # large positive number to ensure x variables with ju indices are indeed unstable!

    class Resources:
        f, fx, v, h = None, None, None, None

    resources = Resources()

    def resourcesfn_r(x, p, c, fopts, hopts, vopts):
        # Precompute here f, fx, v and h, to demonstrate shared resources among functions are correctly precalculated
        cx = c[:nx]
        cp = c[nx:][:np]
        cz = c[nx:][np:][ny:]
        resources.f  = 0.5 * (x.T @ Hxx @ x) + x.T @ Hxp @ p + cx.T @ x
        resources.fx = Hxx @ x + Hxp @ p + cx
        resources.v  = Vpx @ x + Vpp @ p + cp
        resources.h  =  Jx @ x +  Jp @ p + cz

    def objectivefn_f(res, x, p, c, opts):
        res.f   = resources.f  # already computed in the resources function
        res.fx  = resources.fx  # already computed in the resources function
        res.fxx = Hxx
        res.fxp = Hxp
        if opts.eval.fxc:  # compute d(fx)/dc = [d(fx)/d(cx), d(fx)/d(cp), d(fx)/d(cy), d(fx)/d(cz)]
            res.fxc = npy.hstack([Ixx, Oxp, Oxy, Oxz])

    def constraintfn_v(res, x, p, c, opts):
        res.val = resources.v  # already computed in the resources function
        res.ddx = Vpx
        res.ddp = Vpp
        if opts.eval.ddc:  # compute dv/dc = [dv/d(cx), dv/d(cp), dv/d(cy), dv/d(cz)]
            res.ddc = npy.hstack([Opx, Ipp, Opy, Opz])

    def constraintfn_h(res, x, p, c, opts):
        res.val = resources.h  # already computed in the resources function
        res.ddx = Jx
        res.ddp = Jp
        if opts.eval.ddc:  # compute dv/dc = [dv/d(cx), dv/d(cp), dv/d(cy), dv/d(cz)]
            res.ddc = npy.hstack([Ozx, Ozp, Ozy, Izz])

    xlower = npy.full(nx, -npy.inf)
    xupper = npy.full(nx,  npy.inf)

    xlower[jul] = 1.5  # this should be greater than 1.0
    xupper[jul] = 2.0  # this should be greater than xlower[jul]

    xlower[juu] = 0.0  # this should be less than xupper[juu]
    xupper[juu] = 0.5  # this should be less than 1.0

    dims = Dims()
    dims.x  = nx
    dims.p  = np
    dims.be = ny
    dims.he = nz
    dims.c  = nc

    problem = Problem(dims)
    problem.r = resourcesfn_r
    problem.f = objectivefn_f
    problem.he = constraintfn_h
    problem.v = constraintfn_v
    problem.Aex = Ax
    problem.Aep = Ap
    problem.be = cy
    problem.xlower = xlower
    problem.xupper = xupper
    problem.plower = npy.full(np, -npy.inf)
    problem.pupper = npy.full(np,  npy.inf)
    problem.c = npy.concatenate([cx, cp, cy, cz])  # set here the sensitive parameters in the optimization problem
    problem.bec = npy.hstack([Oyx, Oyp, Iyy, Oyz])

    options = Options()
    options.output.active = True
    options.output.filename = f"output-solver-nx={nx}-np={np}-ny={ny}-nz={nz}-nl={nl}-nul={nul}-nuu={nuu}-diagHxx={diagHxx}.txt"

    options.newtonstep.linearsolver.method = \
        LinearSolverMethod.Rangespace if diagHxx else \
        LinearSolverMethod.Nullspace

    solver = Solver()
    solver.setOptions(options)

    state = State(dims)

    res = solver.solve(problem, state)

    if not res.succeeded:
        print(f"Check output-solver-nx={nx}-np={np}-ny={ny}-nz={nz}-nl={nl}-nul={nul}-nuu={nuu}-diagHxx={diagHxx}.txt")

    assert res.succeeded

    sensitivity = Sensitivity()

    res = solver.solve(problem, state, sensitivity)

    assert res.succeeded

    bec = problem.bec
    bec[ny - nl:] = 0.0  # set last nl rows to zero, otherwise the tests with linearly dependent rows do not pass

    xc = sensitivity.xc
    pc = sensitivity.pc

    assert_array_almost_equal(Ax @ xc + Ap @ pc, bec)

