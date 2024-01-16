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


tested_nx = [5, 10, 20, 50]  # The tested number of x variables
tested_np = [0, 5, 10]       # The tested number of p variables
tested_ny = [2, 5, 10]       # The tested number of y variables
tested_nz = [0, 2, 5]        # The tested number of z variables
tested_nl = [0, 1, 2]        # The tested number of linearly dependent rows in Ax

@pytest.mark.parametrize("nx", tested_nx)
@pytest.mark.parametrize("np", tested_np)
@pytest.mark.parametrize("ny", tested_ny)
@pytest.mark.parametrize("nz", tested_nz)
@pytest.mark.parametrize("nl", tested_nl)
def testMatrixRWQ(nx, np, ny, nz, nl):

    params = MasterParams(nx, np, ny, nz, nl)

    if params.invalid(): return

    dims = params.dims

    W = createMatrixViewW(params)

    weights = npy.ones(dims.nx)

    echelonizerW = EchelonizerW()
    echelonizerW.initialize(dims, W.Ax, W.Ap)
    echelonizerW.update(W.Jx, W.Jp, weights)

    RWQ = echelonizerW.RWQ()

    R   = RWQ.R
    Sbn = RWQ.Sbn
    Sbp = RWQ.Sbp
    jb  = RWQ.jb
    jn  = RWQ.jn

    nb = len(jb)

    Ibb = npy.eye(nb)

    assert_almost_equal(Ibb, R @ W.Wx[:, jb])
    assert_almost_equal(Sbn, R @ W.Wx[:, jn])
    assert_almost_equal(Sbp, R @ W.Wp)
