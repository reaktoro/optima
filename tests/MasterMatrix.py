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
def testMasterMatrix(nx, np, ny, nz, nl, nu, diagHxx):

    params = MasterParams(nx, np, ny, nz, nl, nu, diagHxx)

    if params.invalid(): return

    nw = params.dims.nw

    M = createMasterMatrix(params)

    #==========================================================================
    # Check method MasterMatrix::operator Matrix()
    #==========================================================================

    Opw = npy.zeros((np, nw))
    Oww = npy.zeros((nw, nw))

    Mmat = npy.block([
        [ M.H.Hxx, M.H.Hxp, M.W.Wx.T],
        [ M.V.Vpx, M.V.Vpp, Opw],
        [ M.W.Wx , M.W.Wp , Oww],
    ])

    ju = M.ju

    Mmat[:, ju]  = 0.0
    Mmat[ju, :]  = 0.0
    Mmat[ju, ju] = 1.0

    assert npy.all(Mmat == M.array())
