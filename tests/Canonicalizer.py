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
tested_ny      = [2, 5]         # The tested number of y variables
tested_nz      = [0, 5]         # The tested number of z variables
tested_nl      = [0, 2]         # The tested number of linearly dependent rows in Ax
tested_nu      = [0, 2]         # The tested number of unstable variables
tested_diagHxx = [False, True]  # The tested options for Hxx structure

@pytest.mark.parametrize("nx"     , tested_nx)
@pytest.mark.parametrize("np"     , tested_np)
@pytest.mark.parametrize("ny"     , tested_ny)
@pytest.mark.parametrize("nz"     , tested_nz)
@pytest.mark.parametrize("nl"     , tested_nl)
@pytest.mark.parametrize("nu"     , tested_nu)
@pytest.mark.parametrize("diagHxx", tested_diagHxx)
def testCanonicalizer(nx, np, ny, nz, nl, nu, diagHxx):

    params = MasterParams(nx, np, ny, nz, nl, nu, diagHxx)

    if params.invalid(): return

    M = createMasterMatrix(params)

    canonicalizer = Canonicalizer(M)

    Mc = canonicalizer.canonicalMatrix()

    H   = M.H
    V   = M.V
    W   = M.W
    RWQ = M.RWQ

    #==========================================================================
    # Check jb, jn, js, ju in the canonical form of master matrix
    #==========================================================================

    # Assert canonical forms of W and M have the same basic and non-basic variables
    assert set(Mc.jb) == set(RWQ.jb)
    assert set(Mc.jn) == set(RWQ.jn)

    # Assert the unstable variables are non-basic only
    assert set(Mc.ju).issubset(set(RWQ.jn))

    # Assert js and ju are disjoint and its union comprehends all variables
    assert set(Mc.js) & set(Mc.ju) == set()
    assert set(Mc.js) | set(Mc.ju) == set(range(nx))

    # Assert js and ju are disjoint and its union comprehends all variables
    assert set(Mc.js) == set(M.js)
    assert set(Mc.ju) == set(M.ju)

    #==========================================================================
    # Check Hss, Hsp, Vps, Vpp in the canonical form of master matrix
    #==========================================================================
    js = Mc.js
    ju = Mc.ju

    assert npy.all(Mc.Hss == H.Hxx[:, js][js, :])
    assert npy.all(Mc.Hsp == H.Hxp[js, :])

    assert npy.all(Mc.Vps == V.Vpx[:, js])
    assert npy.all(Mc.Vpp == V.Vpp)

    #==========================================================================
    # Check Sbn, Sbp, R in the canonical form of master matrix
    #--------------------------------------------------------------------------
    # Tests below are based on the fact that Rb*W*Q = [Ibb Sbn Sbp] where
    # Q = (jb, jn) and R = [Rb; Rl], Rl is associated with linearly
    # dependent rows in Wx.
    #==========================================================================
    nb  = Mc.dims.nb
    nbs = Mc.dims.nbs
    nns = Mc.dims.nns

    jb  = Mc.jb
    jn  = Mc.jn
    jbs = jb[:nbs]
    jns = jn[:nns]

    Rbs = Mc.Rbs

    Ibsbs = npy.eye(nbs)
    Sbsns = Mc.Sbsns
    Sbsp  = Mc.Sbsp

    assert_almost_equal(Ibsbs, Rbs @ W.Wx[:, jbs])
    assert_almost_equal(Sbsns, Rbs @ W.Wx[:, jns])
    assert_almost_equal(Sbsp,  Rbs @ W.Wp)

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

    dims = Mc.dims

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
