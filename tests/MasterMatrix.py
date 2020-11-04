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

tested_nx  = [5, 10, 20, 50]  # The tested number of x variables
tested_np  = [0, 5, 10]       # The tested number of p variables
tested_ny  = [2, 5, 10]       # The tested number of y variables
tested_nz  = [0, 2, 5]        # The tested number of z variables
tested_nl  = [0, 1, 2]        # The tested number of linearly dependent rows in Ax
tested_nu  = [0, 1, 2]        # The tested number of unstable variables

@mark.parametrize("nx",  tested_nx)
@mark.parametrize("np",  tested_np)
@mark.parametrize("ny",  tested_ny)
@mark.parametrize("nz",  tested_nz)
@mark.parametrize("nl",  tested_nl)
@mark.parametrize("nu",  tested_nu)
def testMasterMatrix(nx, np, ny, nz, nl, nu):

    nw = ny + nz

    # Ensure nx is larger than np and nw
    if nx < np or nx < nw: return

    # Ensure nl < ny
    if ny <= nl: return

    Hxx = random.rand(nx, nx)
    Hxp = random.rand(nx, np)

    isHxxDiag = False

    Vpx = random.rand(np, nx)
    Vpp = random.rand(np, np)

    Wx = random.rand(nw, nx)
    Wp = random.rand(nw, np)

    H = MatrixViewH(Hxx, Hxp, isHxxDiag)
    V = MatrixViewV(Vpx, Vpp)
    W = MatrixViewW(Wx, Wp)

    assert all(H.Hxx == Hxx)
    assert all(H.Hxp == Hxp)

    assert H.isHxxDiag == isHxxDiag

    assert all(V.Vpx == Vpx)
    assert all(V.Vpp == Vpp)

    assert all(W.Wx == Wx)
    assert all(W.Wp == Wp)


    # W = createMasterMatrixViewW(nx, np, ny, nz, nl)
    # H = createMasterMatrixViewH(nx, np)
    # V = createMasterMatrixViewV(nx, np)

    # Wbar = W.echelonForm()

    # ju = Wbar.jn[:nu]  # the first nu` non-basic variables are considered unstable

    # M = MasterMatrix(nx, np, ny, nz)

    # M.update(H, V, W, ju)

    # Mbar = M.canonicalForm()

    # #==========================================================================
    # # Check jb, jn, js, ju in the canonical form of master matrix
    # #==========================================================================

    # # Assert canonical forms of W and M have the same basic and non-basic variables
    # assert set(Mbar.jb) == set(Wbar.jb)
    # assert set(Mbar.jn) == set(Wbar.jn)

    # assert set(Mbar.js) == set(range(nx)) - set(ju)
    # assert set(Mbar.ju) == set(ju)

    # #==========================================================================
    # # Check Hss, Hsp, Vps, Vpp in the canonical form of master matrix
    # #==========================================================================

    # js = Mbar.js
    # ju = Mbar.ju

    # assert all(Mbar.Hss == H.Hxx[:, js][js, :])
    # assert all(Mbar.Hsp == H.Hxp[js, :])

    # assert all(Mbar.Vps == V.Vpx[:, js])
    # assert all(Mbar.Vpp == V.Vpp)

    # #==========================================================================
    # # Check Sbn, Sbp, R in the canonical form of master matrix
    # #--------------------------------------------------------------------------
    # # Tests below are based on the fact that Rb*W*Q = [Ibb Sbn Sbp] where
    # # Q = (jb, jn) and R = [Rb; Rl], Rl is associated with linearly
    # # dependent rows in Wx.
    # #==========================================================================

    # jb = Mbar.jb
    # jn = Mbar.jn

    # nb = len(jb)
    # Rb = Mbar.R[:nb, :]

    # Ibb = eye(nb)
    # Sbn = Mbar.Sbn
    # Sbp = Mbar.Sbp

    # assert_almost_equal(Ibb, Rb @ W.Wx[:, jb])
    # assert_almost_equal(Sbn, Rb @ W.Wx[:, jn])
    # assert_almost_equal(Sbp, Rb @ W.Wp)

    # #==========================================================================
    # # Check matrix views in the canonical form of master matrix
    # #==========================================================================

    # assert all(Mbar.Ws == W.Wx[:, js])
    # assert all(Mbar.Wu == W.Wx[:, ju])
    # assert all(Mbar.Wp == W.Wp)
    # assert all(Mbar.As == W.Ax[:, js])
    # assert all(Mbar.Au == W.Ax[:, ju])
    # assert all(Mbar.Ap == W.Ap)
    # assert all(Mbar.Js == W.Jx[:, js])
    # assert all(Mbar.Ju == W.Jx[:, ju])
    # assert all(Mbar.Jp == W.Jp)

    # #==========================================================================
    # # Check dimension variables in the canonical form of master matrix
    # #==========================================================================

    # # Compute expected dimension variables
    # nw = ny + nz

    # nu = len(ju)
    # ns = nx - nu

    # nb = len(jb)
    # nn = len(jn)
    # nl = nw - nb

    # nbs = len(set(jb) - set(ju))
    # nbu = len(set(jb) & set(ju))
    # nns = len(set(jn) - set(ju))
    # nnu = len(set(jn) & set(ju))

    # jbs = jb[:nbs]
    # jns = jn[:nns]

    # nbe = len([i for i in range(nbs) if abs(H.Hxx[jbs[i], jbs[i]]) >= max(1.0, abs(V.Vpx[:, jbs[i]]).max(initial=0.0))])
    # nne = len([i for i in range(nns) if abs(H.Hxx[jns[i], jns[i]]) >= max(abs(Sbn[:, i]).max(initial=0.0), abs(V.Vpx[:, jns[i]]).max(initial=0.0))])

    # nbi = nbs - nbe
    # nni = nns - nne

    # dims = Mbar.dims

    # assert dims.nx == nx
    # assert dims.np == np
    # assert dims.ny == ny
    # assert dims.nz == nz
    # assert dims.nw == nw
    # assert dims.ns == ns
    # assert dims.nu == nu
    # assert dims.nb == nb
    # assert dims.nn == nn
    # assert dims.nl == nl
    # assert dims.nbs == nbs
    # assert dims.nbu == nbu
    # assert dims.nns == nns
    # assert dims.nnu == nnu
    # assert dims.nbe == nbe
    # assert dims.nbi == nbi
    # assert dims.nne == nne
    # assert dims.nni == nni

    # #==========================================================================
    # # Check method MasterMatrix::matrix()
    # #==========================================================================

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

    # assert all(Mmat == M.matrix())
