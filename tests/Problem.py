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

@pytest.mark.parametrize("nx", tested_dims_x)
@pytest.mark.parametrize("np", tested_dims_p)
@pytest.mark.parametrize("mbe", tested_dims_be)
@pytest.mark.parametrize("mbg", tested_dims_bg)
@pytest.mark.parametrize("mhe", tested_dims_he)
@pytest.mark.parametrize("mhg", tested_dims_hg)
def testProblem(nx, np, mbe, mbg, mhe, mhg):

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

    assert_array_equal(problem.Aex, npy.zeros((mbe, nx)))
    assert_array_equal(problem.Aep, npy.zeros((mbe, np)))
    assert_array_equal(problem.Agx, npy.zeros((mbg, nx)))
    assert_array_equal(problem.Agp, npy.zeros((mbg, np)))
    assert_array_equal(problem.be, npy.zeros(mbe))
    assert_array_equal(problem.bg, npy.zeros(mbg))
    assert_array_equal(problem.xlower, npy.ones(nx) * -npy.inf)
    assert_array_equal(problem.xupper, npy.ones(nx) *  npy.inf)
