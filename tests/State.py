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
tested_dims_x = [25]

# The tested number of linear equality constraint equations.
tested_dims_be = list(range(5))

# The tested number of linear inequality constraint equations.
tested_dims_bg = list(range(5))

# The tested number of non-linear equality constraint equations.
tested_dims_he = list(range(5))

# The tested number of non-linear inequality constraint equations.
tested_dims_hg = list(range(5))

# Combination of all tested cases
testdata = product(
    tested_dims_x,
    tested_dims_be,
    tested_dims_bg,
    tested_dims_he,
    tested_dims_hg
)

@mark.parametrize("args", testdata)
def test_state(args):

    nx, mbe, mbg, mhe, mhg = args

    n = nx + mbg + mhg
    m = mbe + mbg + mhe + mhg

    dims = Dims()
    dims.x = nx
    dims.be = mbe
    dims.bg = mbg
    dims.he = mhe
    dims.hg = mhg

    state = State(dims)

    assert len(state.x) == nx
    assert len(state.y) == m
    assert len(state.ybe) == mbe
    assert len(state.ybg) == mbg
    assert len(state.yhe) == mhe
    assert len(state.yhg) == mhg
    assert len(state.z) == nx
    assert len(state.xbar) == n
    assert len(state.zbar) == n
    assert len(state.xbg) == mbg
    assert len(state.xhg) == mhg

    assert allclose(state.xbar, zeros(n))
    assert allclose(state.zbar, zeros(n))
    assert allclose(state.y, zeros(m))

    state.xbar = linspace(1.0, n, n)
    state.zbar = linspace(1.0, n, n) * 3
    state.y    = linspace(1.0, m, m) * 5

    assert allclose(state.x  , state.xbar[:nx])
    assert allclose(state.ybe, state.y[:mbe])
    assert allclose(state.ybg, state.y[mbe:][:mbg])
    assert allclose(state.yhe, state.y[mbe:][mbg:][:mhe])
    assert allclose(state.yhg, state.y[mbe:][mbg:][mhe:])
    assert allclose(state.z  , state.zbar[:nx])
    assert allclose(state.xbg, state.xbar[nx:][:mbg])
    assert allclose(state.xhg, state.xbar[nx:][mbg:])

