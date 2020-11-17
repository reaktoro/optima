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


from testing.optima import *
from testing.utils.matrices import *


# The tested number of variables in x.
tested_dims_x = [10]

# The tested number of variables in p.
tested_dims_p = [0, 5]

# The tested number of linear equality constraint equations.
tested_dims_be = list(range(2))

# The tested number of linear inequality constraint equations.
tested_dims_bg = list(range(2))

# The tested number of non-linear equality constraint equations.
tested_dims_he = list(range(2))

# The tested number of non-linear inequality constraint equations.
tested_dims_hg = list(range(2))

@pytest.mark.parametrize("nx" , tested_dims_x)
@pytest.mark.parametrize("np" , tested_dims_p)
@pytest.mark.parametrize("nbe", tested_dims_be)
@pytest.mark.parametrize("nbg", tested_dims_bg)
@pytest.mark.parametrize("nhe", tested_dims_he)
@pytest.mark.parametrize("nhg", tested_dims_hg)
def testState(nx, np, nbe, nbg, nhe, nhg):

    nxbar = nx + nbg + nhg
    ny = nbe + nbg
    nz = nhe + nhg

    dims = Dims()
    dims.x = nx
    dims.p = np
    dims.be = nbe
    dims.bg = nbg
    dims.he = nhe
    dims.hg = nhg

    state = State(dims)

    assert len(state.x) == nx
    assert len(state.p) == np
    assert len(state.y) == ny
    assert len(state.ye) == nbe
    assert len(state.yg) == nbg
    assert len(state.z) == nz
    assert len(state.ze) == nhe
    assert len(state.zg) == nhg
    assert len(state.s) == nx
    assert len(state.xbar) == nxbar
    assert len(state.sbar) == nxbar
    assert len(state.xbg) == nbg
    assert len(state.xhg) == nhg

    assert_array_equal(state.x, npy.zeros(nx))
    assert_array_equal(state.p, npy.zeros(np))
    assert_array_equal(state.y, npy.zeros(ny))
    assert_array_equal(state.z, npy.zeros(nz))
    assert_array_equal(state.xbar, npy.zeros(nxbar))
    assert_array_equal(state.sbar, npy.zeros(nxbar))

    state.xbar = npy.linspace(1.0, nxbar, nxbar)
    state.sbar = npy.linspace(1.0, nxbar, nxbar) * 3
    state.y    = npy.linspace(1.0, ny, ny) * 5
    state.z    = npy.linspace(1.0, nz, nz) * 7

    assert_array_equal(state.x  , state.xbar[:nx])
    assert_array_equal(state.ye , state.w[:nbe])
    assert_array_equal(state.yg , state.w[nbe:][:nbg])
    assert_array_equal(state.ze , state.w[nbe:][nbg:][:nhe])
    assert_array_equal(state.zg , state.w[nbe:][nbg:][nhe:])
    assert_array_equal(state.s  , state.sbar[:nx])
    assert_array_equal(state.xbg, state.xbar[nx:][:nbg])
    assert_array_equal(state.xhg, state.xbar[nx:][nbg:])

