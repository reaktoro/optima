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


tested_nx = [4]     # The tested number of x variables
tested_np = [0, 1]  # The tested number of p variables
tested_ny = [2]     # The tested number of y variables
tested_nz = [0, 1]  # The tested number of z variables
tested_nl = [0, 1]  # The tested number of linearly dependent rows in Ax
tested_nu = [0, 2]  # The tested number of unstable variables

tested_b1 = [True, False]  # The tested value of ObjectiveResult::diagfxx for f(x, p)
tested_b2 = [True, False]  # The tested value of ObjectiveResult::fxx4basicvars for f(x, p)
tested_b3 = [True, False]  # The tested value of ObjectiveResult::succeeded for f(x, p)
tested_b4 = [True, False]  # The tested value of ConstraintResult::ddx4basicvars for h(x, p)
tested_b5 = [True, False]  # The tested value of ConstraintResult::succeeded for h(x, p)
tested_b6 = [True, False]  # The tested value of ConstraintResult::ddx4basicvars for v(x, p)
tested_b7 = [True, False]  # The tested value of ConstraintResult::succeeded for v(x, p)

@pytest.mark.parametrize("nx", tested_nx)
@pytest.mark.parametrize("np", tested_np)
@pytest.mark.parametrize("ny", tested_ny)
@pytest.mark.parametrize("nz", tested_nz)
@pytest.mark.parametrize("nl", tested_nl)
@pytest.mark.parametrize("nu", tested_nu)
@pytest.mark.parametrize("b1", tested_b1)
@pytest.mark.parametrize("b2", tested_b2)
@pytest.mark.parametrize("b3", tested_b3)
@pytest.mark.parametrize("b4", tested_b4)
@pytest.mark.parametrize("b5", tested_b5)
@pytest.mark.parametrize("b6", tested_b6)
@pytest.mark.parametrize("b7", tested_b7)
def testResidualFunction(nx, np, ny, nz, nl, nu, b1, b2, b3, b4, b5, b6, b7):

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

    dims = params.dims

    cx = rng.rand(dims.nx)
    cp = rng.rand(dims.np)
    cz = rng.rand(dims.nz)

    cx[ju] = +1.0e4  # large positive number to ensure the variables with ju indices are indeed unstable!

    def objectivefn_f(res, x, p, c, opts):
        res.f   = 0.5 * (x.T @ Hxx @ x) + x.T @ Hxp @ p + cx.T @ x
        res.fx  = Hxx @ x + Hxp @ p + cx
        res.fxx = Hxx
        res.fxp = Hxp
        res.diagfxx = b1
        res.fxx4basicvars = b2
        res.succeeded = b3

    def constraintfn_h(res, x, p, c, opts):
        res.val = Jx @ x + Jp @ p + cz
        res.ddx = Jx
        res.ddp = Jp
        res.ddx4basicvars = b4
        res.succeeded = b5

    def constraintfn_v(res, x, p, c, opts):
        res.val = Vpx @ x + Vpp @ p + cp
        res.ddx = Vpx
        res.ddp = Vpp
        res.ddx4basicvars = b6
        res.succeeded = b7

    problem = MasterProblem()
    problem.dims = dims
    problem.f = objectivefn_f
    problem.h = constraintfn_h
    problem.v = constraintfn_v
    problem.Ax = Ax
    problem.Ap = Ap
    problem.b = rng.rand(dims.ny)
    problem.xlower = -abs(rng.rand(dims.nx))
    problem.xupper =  abs(rng.rand(dims.nx))
    problem.phi = None

    dims = params.dims

    u = MasterVector(dims)
    x = u.x = (problem.xlower + problem.xupper) * 0.5
    p = u.p = rng.rand(dims.np)
    w = u.w = rng.rand(dims.nw)

    u.x[ju] = problem.xlower[ju]  # ensure the unstable variables are attached to lower bounds

    F = ResidualFunction()
    F.initialize(problem)

    status = F.update(u)

    result = F.result()

    x, p, w = u.x, u.p, u.w

    assert result.f.f             == approx(0.5 * (x.T @ Hxx @ x) + x.T @ Hxp @ p + cx.T @ x)
    assert result.f.fx            == approx(Hxx @ x + Hxp @ p + cx)
    assert result.f.fxx           == approx(Hxx)
    assert result.f.fxp           == approx(Hxp)
    assert result.f.diagfxx       == b1
    assert result.f.fxx4basicvars == b2
    assert result.f.succeeded     == b3

    if nz > 0:
        assert result.h.val           == approx(Jx @ x + Jp @ p + cz)
        assert result.h.ddx           == approx(Jx)
        assert result.h.ddp           == approx(Jp)
        assert result.h.ddx4basicvars == b4
        assert result.h.succeeded     == b5

    if np > 0:
        assert result.v.val           == approx(Vpx @ x + Vpp @ p + cp)
        assert result.v.ddx           == approx(Vpx)
        assert result.v.ddp           == approx(Vpp)
        assert result.v.ddx4basicvars == b6
        assert result.v.succeeded     == b7

    assert result.succeeded == all([
        result.f.succeeded,
        result.h.succeeded,
        result.v.succeeded
    ])

    if not result.succeeded:
        return

    Jm = result.Jm
    Jc = result.Jc

    assert set(M.js) == set(Jm.js)
    assert set(M.ju) == set(Jm.ju)

    jb  = Jm.RWQ.jb
    jn  = Jm.RWQ.jn
    Sbn = Jm.RWQ.Sbn

    g = result.f.fx

    assert result.stabilitystatus.s == approx(g + Wx.T @ w)

