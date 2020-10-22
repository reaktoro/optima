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
from numpy.testing import assert_allclose, assert_almost_equal
from pytest import approx, mark

tested_nx = [5, 10, 20, 50]  # The tested number of x variables
tested_np = [0, 5, 10]       # The tested number of p variables

@mark.parametrize("nx", tested_nx)
@mark.parametrize("np", tested_np)
def test_jacobian_block_v(nx, np):

    Vpx = random.rand(np, nx)
    Vpp = random.rand(np, np)

    V = JacobianBlockV(nx, np)

    V.Vpx = Vpx
    V.Vpp = Vpp

    assert (V.Vpx == Vpx).all()
    assert (V.Vpp == Vpp).all()

    V = JacobianBlockV(Vpx, Vpp)

    assert (V.Vpx == Vpx).all()
    assert (V.Vpp == Vpp).all()

