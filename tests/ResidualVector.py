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


tested_nx      = [15, 20]       # The tested number of x variables
tested_np      = [0, 5]         # The tested number of p variables
tested_ny      = [5, 8]         # The tested number of y variables
tested_nz      = [0, 5]         # The tested number of z variables
tested_nl      = [0, 2]         # The tested number of linearly dependent rows in Ax
tested_nu      = [0, 2]         # The tested number of unstable variables

@pytest.mark.parametrize("nx", tested_nx)
@pytest.mark.parametrize("np", tested_np)
@pytest.mark.parametrize("ny", tested_ny)
@pytest.mark.parametrize("nz", tested_nz)
@pytest.mark.parametrize("nl", tested_nl)
@pytest.mark.parametrize("nu", tested_nu)
def testResidualVector(nx, np, ny, nz, nl, nu):

    params = MasterParams(nx, np, ny, nz, nl, nu)

    if params.invalid(): return

    M = createMasterMatrix(params)

    canonicalizer = Canonicalizer(M)

    Mc = canonicalizer.canonicalMatrix()

    Wx, Wp = M.W.Wx, M.W.Wp
    Ax, Jx = npy.vsplit(Wx, [ny])
    Ap, Jp = npy.vsplit(Wp, [ny])

    dims = params.dims

    F = ResidualVector()

    x = rng.rand(nx)
    p = rng.rand(np)
    y = rng.rand(ny)
    z = rng.rand(nz)
    g = rng.rand(nx)
    v = rng.rand(np)
    b = rng.rand(ny)
    h = rng.rand(nz)

    F.update(Mc, Wx, Wp, x, p, y, z, g, v, b, h)

    nbs = Mc.dims.nbs

    js, ju  = Mc.js, Mc.ju

    As = Ax[:, js]
    Au = Ax[:, ju]
    Js = Jx[:, js]

    Rbs = Mc.Rbs

    ax = npy.zeros(nx)
    ax[js] = -(g[js] + As.T @ y + Js.T @ z)

    ap = -v
    ay = -(As @ x[js] + Au @ x[ju] + Ap @ p - b)
    az = -h
    aw = npy.concatenate([ay, az])
    awbs = Rbs @ aw

    a = F.masterVector()

    assert_almost_equal(a.x, ax)
    assert_almost_equal(a.p, ap)
    assert_almost_equal(a.w, aw)

    a = F.canonicalVector()

    assert_almost_equal(a.xs, ax[js])
    assert_almost_equal(a.p, ap)
    assert_almost_equal(a.wbs, awbs)
