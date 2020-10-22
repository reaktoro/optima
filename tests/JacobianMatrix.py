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
from numpy.testing import assert_allclose, assert_almost_equal
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
def test_jacobian_matrix(nx, np, ny, nz, nl, nbu, ju):

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

    Wbar = W.canonicalForm()
    Mbar = M.canonicalForm()

    #==========================================================================
    # Check jb, jn, js, ju in the canonical form of Jacobian matrix
    #==========================================================================

    # Assert canonical forms of W and M have the same basic and non-basic variables
    assert set(Mbar.jb) == set(Wbar.jb)
    assert set(Mbar.jn) == set(Wbar.jn)

    assert set(Mbar.js) == set(range(nx)) - set(ju)
    assert set(Mbar.ju) == set(ju)

    #==========================================================================
    # Check Hss, Hsp, Vps, Vpp in the canonical form of Jacobian matrix
    #==========================================================================

    js = Mbar.js
    ju = Mbar.ju

    assert all(Mbar.Hss == Hxx[:, js][js, :])
    assert all(Mbar.Hsp == Hxp[js, :])

    assert all(Mbar.Vps == Vpx[:, js])
    assert all(Mbar.Vpp == Vpp)

    #==========================================================================
    # Check Sbn, Sbp, R in the canonical form of Jacobian matrix
    #--------------------------------------------------------------------------
    # Tests below are based on the fact that Rb*W*Q = [Ibb Sbn Sbp] where
    # Q = (jb, jn) and R = [Rb; Rl], Rl is associated with linearly
    # dependent rows in Wx.
    #==========================================================================

    jb = Mbar.jb
    jn = Mbar.jn

    nb = len(jb)
    Rb = Mbar.R[:nb, :]

    Ibb = eye(nb)
    Sbn = Mbar.Sbn
    Sbp = Mbar.Sbp

    assert_almost_equal(Ibb, Rb @ W.Wx[:, jb])
    assert_almost_equal(Sbn, Rb @ W.Wx[:, jn])
    assert_almost_equal(Sbp, Rb @ W.Wp)

    #==========================================================================
    # Check matrix views in the canonical form of Jacobian matrix
    #==========================================================================

    assert all(Mbar.Ws == W.Wx[:, js])
    assert all(Mbar.Wu == W.Wx[:, ju])
    assert all(Mbar.Wp == W.Wp)
    assert all(Mbar.As == Ax[:, js])
    assert all(Mbar.Au == Ax[:, ju])
    assert all(Mbar.Ap == Ap)
    assert all(Mbar.Js == Jx[:, js])
    assert all(Mbar.Ju == Jx[:, ju])
    assert all(Mbar.Jp == Jp)

    #==========================================================================
    # Check dimension variables in the canonical form of Jacobian matrix
    #==========================================================================

    # Compute expected dimension variables to test method JacobianMatrix::dims()
    nw = ny + nz

    nu = len(ju)
    ns = nx - nu

    nb = len(jb)
    nn = len(jn)
    nl = nw - nb

    nbs = len(set(jb) - set(ju))
    nbu = len(set(jb) & set(ju))
    nns = len(set(jn) - set(ju))
    nnu = len(set(jn) & set(ju))

    jbs = jb[:nbs]
    jns = jn[:nns]

    nbe = len([i for i in range(nbs) if abs(Hxx[jbs[i], jbs[i]]) >= max(1.0, abs(Vpx[:, jbs[i]]).max(initial=0.0))])
    nne = len([i for i in range(nns) if abs(Hxx[jns[i], jns[i]]) >= max(abs(Sbn[:, i]).max(initial=0.0), abs(Vpx[:, jns[i]]).max(initial=0.0))])

    nbi = nbs - nbe
    nni = nns - nne

    dims = M.dims()

    assert dims.nx == nx
    assert dims.np == np
    assert dims.ny == ny
    assert dims.nz == nz
    assert dims.nw == nw
    assert dims.ns == ns
    assert dims.nu == nu
    assert dims.nb == nb
    assert dims.nn == nn
    assert dims.nl == nl
    assert dims.nbs == nbs
    assert dims.nbu == nbu
    assert dims.nns == nns
    assert dims.nnu == nnu
    assert dims.nbe == nbe
    assert dims.nbi == nbi
    assert dims.nne == nne
    assert dims.nni == nni
