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
from pytest import approx, mark
from itertools import product


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

# Combination of all tested cases
testdata = product(
    tested_dims_x,
    tested_dims_p,
    tested_dims_be,
    tested_dims_bg,
    tested_dims_he,
    tested_dims_hg
)

@mark.parametrize("args", testdata)
def test_state(args):

    nx, np, nbe, nbg, nhe, nhg = args

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

    assert allclose(state.x, zeros(nx))
    assert allclose(state.p, zeros(np))
    assert allclose(state.y, zeros(ny))
    assert allclose(state.z, zeros(nz))
    assert allclose(state.xbar, zeros(nxbar))
    assert allclose(state.sbar, zeros(nxbar))

    state.xbar = linspace(1.0, nxbar, nxbar)
    state.sbar = linspace(1.0, nxbar, nxbar) * 3
    state.y    = linspace(1.0, ny, ny) * 5
    state.z    = linspace(1.0, nz, nz) * 7

    assert allclose(state.x  , state.xbar[:nx])
    assert allclose(state.ye , state.y[:nbe])
    assert allclose(state.yg , state.y[nbe:][:nbg])
    assert allclose(state.ze , state.y[nbe:][nbg:][:nhe])
    assert allclose(state.zg , state.y[nbe:][nbg:][nhe:])
    assert allclose(state.s  , state.sbar[:nx])
    assert allclose(state.xbg, state.xbar[nx:][:nbg])
    assert allclose(state.xhg, state.xbar[nx:][nbg:])

