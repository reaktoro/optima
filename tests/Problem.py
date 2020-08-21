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
def test_problem(args):

    nx, np, mbe, mbg, mhe, mhg = args

    n = nx + mbg + mhg
    m = mbe + mbg + mhe + mhg

    dims = Dims()
    dims.x = nx
    dims.p = np
    dims.be = mbe
    dims.bg = mbg
    dims.he = mhe
    dims.hg = mhg

    problem = Problem(dims)

    assert allclose(problem.Aex, zeros((mbe, nx)))
    assert allclose(problem.Aep, zeros((mbe, np)))
    assert allclose(problem.Agx, zeros((mbg, nx)))
    assert allclose(problem.Agp, zeros((mbg, np)))
    assert allclose(problem.be, zeros(mbe))
    assert allclose(problem.bg, zeros(mbg))
    assert allclose(problem.xlower, ones(nx) * -inf)
    assert allclose(problem.xupper, ones(nx) *  inf)
