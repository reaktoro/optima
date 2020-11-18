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

    assert u.x.flags["WRITEABLE"] == True  # ensure u.x[indices] = xsubvec is working - need to use def_property for x, and not def_readwrite!
    assert u.p.flags["WRITEABLE"] == True  # ensure u.p[indices] = psubvec is working - need to use def_property for p, and not def_readwrite!
    assert u.w.flags["WRITEABLE"] == True  # ensure u.w[indices] = wsubvec is working - need to use def_property for w, and not def_readwrite!

