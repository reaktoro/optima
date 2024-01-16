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


tested_nx = [10, 20]  # The tested number of x variables
tested_np = [0, 5]    # The tested number of p variables
tested_ny = [5, 8]    # The tested number of y variables
tested_nz = [0, 5]    # The tested number of z variables
tested_nl = [0, 2]    # The tested number of linearly dependent rows in Ax
tested_nu = [0, 3]    # The tested number of unstable variables

@pytest.mark.parametrize("nx", tested_nx)
@pytest.mark.parametrize("np", tested_np)
@pytest.mark.parametrize("ny", tested_ny)
@pytest.mark.parametrize("nz", tested_nz)
@pytest.mark.parametrize("nl", tested_nl)
@pytest.mark.parametrize("nu", tested_nu)
def testResidualFunction(nx, np, ny, nz, nl, nu):

    params = MasterParams(nx, np, ny, nz, nl, nu)

    if params.invalid(): return

    M = createMasterMatrix(params)

    Hxx = M.H.Hxx
    Hxp = M.H.Hxp
    Vpx = M.V.Vpx
    Vpp = M.V.Vpp
    Wx  = M.W.Wx
    Wp  = M.W.Wp
    Ax  = M.W.Ax
    Ap  = M.W.Ap
    Jx  = M.W.Jx
    Jp  = M.W.Jp
    ju  = M.ju
    js  = M.js

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

    dims = params.dims

    nc = nx + np + ny + nz

    c = rng.rand(nc)  # c = [cx, cp, cy, cz] -- the parameters used for sensitivity derivative calculations

    c[:nx][ju] = +1.0e4  # large positive number to ensure x variables with ju indices are indeed unstable!

    def objectivefn_f(res, x, p, c, opts):
        cx = c[:nx]
        res.f   = 0.5 * (x.T @ Hxx @ x) + x.T @ Hxp @ p + cx.T @ x
        res.fx  = Hxx @ x + Hxp @ p + cx
        res.fxx = Hxx
        res.fxp = Hxp
        if opts.eval.fxc:  # compute d(fx)/dc = [d(fx)/d(cx), d(fx)/d(cp), d(fx)/d(cy), d(fx)/d(cz)]
            res.fxc = npy.hstack([Ixx, Oxp, Oxy, Oxz])

    def constraintfn_v(res, x, p, c, opts):
        cp = c[nx:][:np]
        res.val = Vpx @ x + Vpp @ p + cp
        res.ddx = Vpx
        res.ddp = Vpp
        if opts.eval.ddc:  # compute dv/dc = [dv/d(cx), dv/d(cp), dv/d(cy), dv/d(cz)]
            res.ddc = npy.hstack([Opx, Ipp, Opy, Opz])

    def constraintfn_h(res, x, p, c, opts):
        cz = c[nx:][np:][ny:]
        res.val = Jx @ x + Jp @ p + cz
        res.ddx = Jx
        res.ddp = Jp
        if opts.eval.ddc:  # compute dv/dc = [dv/d(cx), dv/d(cp), dv/d(cy), dv/d(cz)]
            res.ddc = npy.hstack([Ozx, Ozp, Ozy, Izz])

    problem = MasterProblem()
    problem.dims = dims
    problem.f = objectivefn_f
    problem.h = constraintfn_h
    problem.v = constraintfn_v
    problem.Ax = Ax
    problem.Ap = Ap
    problem.b = c[nx:][np:][:ny]
    problem.xlower = -abs(rng.rand(dims.nx))
    problem.xupper =  abs(rng.rand(dims.nx))
    problem.phi = None
    problem.c = c  # set here the sensitive parameters in the master optimization problem
    problem.bc = npy.hstack([Oyx, Oyp, Iyy, Oyz])

    dims = params.dims

    u = MasterVector(dims)
    x = u.x = (problem.xlower + problem.xupper) * 0.5
    p = u.p = rng.rand(dims.np)
    w = u.w = rng.rand(dims.nw)

    u.x[ju] = problem.xlower[ju]  # ensure the unstable variables are attached to lower bounds

    F = ResidualFunction()
    F.initialize(problem)
    F.updateOnlyJacobian(u)  # compute Jacobian matrices d(fx)/dx, d(fx)/dp, d(fx)/dc, dv/dx, dv/dp, dv/dc, dh/dx, dh/dp, dh/dc

    sensitivity = MasterSensitivity()
    state = MasterState(dims)

    ss = SensitivitySolver()

    ss.initialize(problem)
    ss.solve(F, state, sensitivity)

    res = F.result()

    fxc = res.f.fxc
    fxc[ju, :] = 0.0  # ensure rows corresponding to unstable variables are zero

    hc = res.h.ddc
    vc = res.v.ddc

    bc = problem.bc  # the derivatives of b with respect to c
    bc[ny - nl:] = 0.0  # set last nl rows to zero, otherwise the tests with linearly dependent rows do not pass

    rc = npy.vstack([ -fxc, -vc, bc, -hc ])

    xc = sensitivity.xc
    pc = sensitivity.pc
    wc = sensitivity.wc
    sc = sensitivity.sc

    uc = npy.vstack([ xc, pc, wc ])

    Jm = res.Jm.array()

    assert_array_almost_equal(Jm @ uc, rc)
