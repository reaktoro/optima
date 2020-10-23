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
from pytest import mark

tested_nx  = [5, 10, 20, 50]  # The tested number of x variables
tested_np  = [0, 5, 10]       # The tested number of p variables
tested_ny  = [0, 5]           # The tested number of y variables
tested_nz  = [0, 5]           # The tested number of z variables

@mark.parametrize("nx",  tested_nx)
@mark.parametrize("np",  tested_np)
@mark.parametrize("ny",  tested_ny)
@mark.parametrize("nz",  tested_nz)
def test_solution_vector(nx, np, ny, nz):

    u = SolutionVector(nx, np, ny, nz)

    u.x = random.rand(nx)
    u.p = random.rand(np)
    u.y = random.rand(ny)
    u.z = random.rand(nz)

    assert all(u.vec == concatenate([u.x, u.p, u.y, u.z]))

    u.vec = random.rand(len(u.vec))

    assert all(u.vec == concatenate([u.x, u.p, u.y, u.z]))
