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


tested_nx  = [5, 10, 20, 50]  # The tested number of x variables
tested_np  = [0, 5, 10]       # The tested number of p variables
tested_nw  = [5, 8]           # The tested number of w variables

@pytest.mark.parametrize("nx", tested_nx)
@pytest.mark.parametrize("np", tested_np)
@pytest.mark.parametrize("nw", tested_nw)
def testMasterVector(nx, np, nw):

    u = MasterVector(nx, np, nw)

    u.x = random.rand(nx)
    u.p = random.rand(np)
    u.w = random.rand(nw)

    assert all(u.array() == npy.concatenate([u.x, u.p, u.w]))

    u.x[:nx] = 0.0  # ensure u.x[:nx] is a view to actual content, and not a copy
    u.p[:np] = 0.0  # ensure u.p[:np] is a view to actual content, and not a copy
    u.w[:nw] = 0.0  # ensure u.w[:nw] is a view to actual content, and not a copy

    assert all(u.x[:nx] == 0.0)
    assert all(u.p[:np] == 0.0)
    assert all(u.w[:nw] == 0.0)
