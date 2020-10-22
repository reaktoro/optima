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
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from optima import *
from numpy import *
from numpy.testing import assert_almost_equal
from pytest import approx, mark

tested_nx  = [5, 10, 20, 50]  # The tested number of x variables
tested_np  = [0, 5, 10]       # The tested number of p variables
tested_ny  = [2, 5, 10]       # The tested number of y variables
tested_nz  = [0, 2, 5]        # The tested number of z variables
tested_nl  = [0, 1, 2]        # The tested number of linearly dependent rows in Ax
tested_nbu = [0, 1, 2]        # The tested number of basic unstable variables

# The tested indices of unstable variables in x
tested_ju = [
    array([]),
    array([1]),
    array([1, 2, 3])
]

@mark.parametrize("nx",  tested_nx)
@mark.parametrize("np",  tested_np)
@mark.parametrize("ny",  tested_ny)
@mark.parametrize("nz",  tested_nz)
@mark.parametrize("nl",  tested_nl)
@mark.parametrize("nbu", tested_nbu)
@mark.parametrize("ju",  tested_ju)
def test_residual_vector(nx, np, ny, nz, nl, nbu, ju):

    # Ensure nx is larger than np and (ny + nz)
    if nx < np or nx < ny + nz: return

    # Ensure nl < ny
    if ny < nl: return

    Hxx = random.rand(nx, nx)
    Hxp = random.rand(nx, np)
    H = JacobianBlockH(Hxx, Hxp)

    Vpx = random.rand(np, nx)
    Vpp = random.rand(np, np)
    V = JacobianBlockV(Vpx, Vpp)

    Ax = random.rand(ny, nx)
    Ap = random.rand(ny, np)

    Ax[:nl, :] = 0.0  # set last nl rows to be zero so that we have nl linearly dependent rows in Ax

    # Prepare Ax in case there are basic unstable variables.
    if nbu > 0:
        for i in range(nbu):
            if i < len(ju):
                Ax[i, :] = 0.0
                Ax[:, ju[i]] = 1.0  # 1 for the unstable variable, 0 for all others

    W = JacobianBlockW(nx, np, ny, nz, Ax, Ap)

    weights = ones(nx)
    Jx = random.rand(nz, nx)
    Jp = random.rand(nz, np)

    W.update(Jx, Jp, weights)

    M = JacobianMatrix(nx, np, ny, nz)

    M.update(H, V, W, ju)

    F = ResidualVector(nx, np, ny, nz)

    x = random.rand(nx)
    p = random.rand(np)
    y = random.rand(ny)
    z = random.rand(nz)
    g = random.rand(nx)
    v = random.rand(np)
    b = random.rand(ny)
    h = random.rand(nz)

    F.update(M, x, p, y, z, g, v, b, h)

    dims = M.dims()
    nbs = dims.nbs

    Mbar = M.canonicalForm()
    js  = Mbar.js
    ju  = Mbar.ju
    R   = Mbar.R

    ax = -(g + Ax.T @ y + Jx.T @ z)
    ax[ju] = 0.0

    ap = -v
    ay = -(Ax @ x + Ap @ p - b)
    az = -h
    aw = concatenate([ay, az])
    awbar = R @ aw

    Fbar = F.canonicalForm()

    assert_almost_equal(Fbar.axs, ax[js])
    assert_almost_equal(Fbar.ap, ap)
    assert_almost_equal(Fbar.awbs, awbar[:nbs])
