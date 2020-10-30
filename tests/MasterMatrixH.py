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
def test_master_matrix_h(nx, np):

    Hxx = random.rand(nx, nx)
    Hxp = random.rand(nx, np)

    H = MasterMatrixH(nx, np)

    H.Hxx = Hxx
    H.Hxp = Hxp

    assert (H.Hxx == Hxx).all()
    assert (H.Hxp == Hxp).all()

    H = MasterMatrixH(Hxx, Hxp)

    assert (H.Hxx == Hxx).all()
    assert (H.Hxp == Hxp).all()

    assert H.isHxxDiagonal() == False

    H.isHxxDiagonal(True)

    assert H.isHxxDiagonal() == True
