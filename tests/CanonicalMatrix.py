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
from pytest import mark, raises
from utils.matrices import *

tested_nx  = [10]  # The tested number of x variables
tested_np  = [0, 5]       # The tested number of p variables
tested_ny  = [2, 5]       # The tested number of y variables
tested_nz  = [0, 2]        # The tested number of z variables
tested_nl  = [0, 1]        # The tested number of linearly dependent rows in Ax
tested_nu  = [0, 1]        # The tested number of unstable variables

# tested_nx  = [5, 10, 20, 50]  # The tested number of x variables
# tested_np  = [0, 5, 10]       # The tested number of p variables
# tested_ny  = [2, 5, 10]       # The tested number of y variables
# tested_nz  = [0, 2, 5]        # The tested number of z variables
# tested_nl  = [0, 1, 2]        # The tested number of linearly dependent rows in Ax
# tested_nu  = [0, 1, 2]        # The tested number of unstable variables

@mark.parametrize("nx", tested_nx)
@mark.parametrize("np", tested_np)
@mark.parametrize("ny", tested_ny)
@mark.parametrize("nz", tested_nz)
@mark.parametrize("nl", tested_nl)
@mark.parametrize("nu", tested_nu)
def testCanonicalMatrix(nx, np, ny, nz, nl, nu):

    basedims = BaseDims(nx, np, ny, nz)

    set_printoptions(linewidth=10000)

    # Ensure nx is larger than np and (ny + nz)
    if nx < np or nx < ny + nz: return

    # Ensure nl < ny
    if ny <= nl: return

    # # H = createMatrixViewHxxHxp(nx, np)
    # H = createMatrixViewH(nx, np)
    # V = createMatrixViewV(nx, np)
    # W = createMatrixViewW(nx, np, ny, nz, nl)

    # Wbar = createMatrixViewWbar(W)
    # js, ju = createStableUnstableIndices(nu, Wbar)

    # ju = Wbar.jn[:nu]  # the first nu` non-basic variables are considered unstable
    # js = array(set(arange(nx)) - set(ju))

    M = createMasterMatrix(nx, np, ny, nz, nl, nu)

    H = M.H
    V = M.V
    W = M.W

    canonicalmatrix = CanonicalMatrix(basedims)
    canonicalmatrix.update(M)

    Mbar = canonicalmatrix.view()

    #==========================================================================
    # Check jb, jn, js, ju in the canonical form of master matrix
    #==========================================================================

    # Assert canonical forms of W and M have the same basic and non-basic variables
    assert set(Mbar.jb) == set(M.RWQ.jb)
    assert set(Mbar.jn) == set(M.RWQ.jn)

    # Assert the unstable variables are non-basic only
    assert set(Mbar.ju).issubset(set(M.RWQ.jn))

    # Assert js and ju are disjoint and its union comprehends all variables
    assert set(Mbar.js) & set(Mbar.ju) == set()
    assert set(Mbar.js) | set(Mbar.ju) == set(range(nx))

    # Assert js and ju are disjoint and its union comprehends all variables
    assert set(Mbar.js) == set(M.js)
    assert set(Mbar.ju) == set(M.ju)

    #==========================================================================
    # Check Hss, Hsp, Vps, Vpp in the canonical form of master matrix
    #==========================================================================
    js = Mbar.js
    ju = Mbar.ju

    print(f"js = \n{js}")
    print(f"Mbar.Hss = \n{Mbar.Hss}")
    print(f"M.H.Hxx[:, js][js, :] = \n{M.H.Hxx[:, js][js, :]}")
    print(f"M.H.Hxx = \n{M.H.Hxx}")
    assert all(Mbar.Hss == M.H.Hxx[:, js][js, :])
    assert all(Mbar.Hsp == M.H.Hxp[js, :])

    assert all(Mbar.Vps == M.V.Vpx[:, js])
    assert all(Mbar.Vpp == M.V.Vpp)

    #==========================================================================
    # Check Sbn, Sbp, R in the canonical form of master matrix
    #--------------------------------------------------------------------------
    # Tests below are based on the fact that Rb*W*Q = [Ibb Sbn Sbp] where
    # Q = (jb, jn) and R = [Rb; Rl], Rl is associated with linearly
    # dependent rows in Wx.
    #==========================================================================
    nb  = Mbar.dims.nb
    nbs = Mbar.dims.nbs
    nns = Mbar.dims.nns

    jb  = Mbar.jb
    jn  = Mbar.jn
    jbs = jb[:nbs]
    jns = jn[:nns]

    Rbs = Mbar.Rbs

    Ibsbs = eye(nbs)
    Sbsns = Mbar.Sbsns
    Sbsp  = Mbar.Sbsp

    assert_almost_equal(Ibsbs, Rbs @ M.W.Wx[:, jbs])
    assert_almost_equal(Sbsns, Rbs @ M.W.Wx[:, jns])
    assert_almost_equal(Sbsp,  Rbs @ M.W.Wp)

    #==========================================================================
    # Check dimension variables in the canonical form of master matrix
    #==========================================================================

    # Compute expected dimension variables
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

    nbe = len([i for i in range(nbs) if abs(H.Hxx[jbs[i], jbs[i]]) >= max(1.0, abs(V.Vpx[:, jbs[i]]).max(initial=0.0))])
    nne = len([i for i in range(nns) if abs(H.Hxx[jns[i], jns[i]]) >= max(abs(Sbsns[:, i]).max(initial=0.0), abs(V.Vpx[:, jns[i]]).max(initial=0.0))])

    nbi = nbs - nbe
    nni = nns - nne

    dims = Mbar.dims

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

    # ==========================================================================
    # Check method MasterMatrix::matrix()
    # ==========================================================================

    # Hxx = zeros((nx, nx))
    # for i in range(ns):
    #     for j in range(ns):
    #         Hxx[js[i], js[j]] = Mbar.Hss[i, j]
    # Hxx[ju, ju] = 1.0

    # Hxp = zeros((nx, np))
    # Hxp[js, :] = Mbar.Hsp

    # WxT = zeros((nx, nw))
    # WxT[js, :] = Mbar.Ws.T

    # Vpx = zeros((np, nx))
    # Vpx[:, js] = Mbar.Vps

    # Vpp = Mbar.Vpp

    # Wx = zeros((nw, nx))
    # Wx[:, js] = Mbar.Ws

    # Wp = Mbar.Wp

    # Opw = zeros((np, nw))
    # Oww = zeros((nw, nw))

    # Mmat = block([
    #     [Hxx, Hxp, WxT],
    #     [Vpx, Vpp, Opw],
    #     [ Wx,  Wp, Oww],
    # ])

    # assert all(Mmat == M.array())
