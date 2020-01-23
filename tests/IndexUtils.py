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
from pytest import approx


def test_index_utils():
    n = 10

    assert contains([1, 3, 6], 1)
    assert contains([1, 3, 6], 3)
    assert contains([1, 3, 6], 6)
    assert not contains([1, 3, 6], 7)

    inds = arange(n)
    partitionLeft(inds, [1, 3, 5, 7])
    assert set(inds[:4]) == set([1, 3, 5, 7])
    assert set(inds[4:]) == set([0, 2, 4, 6, 8, 9])

    inds = arange(n)
    partitionRight(inds, [1, 3, 5, 7])
    assert set(inds[-4:]) == set([1, 3, 5, 7])
    assert set(inds[:-4]) == set([0, 2, 4, 6, 8, 9])

    inds = arange(n)
    partitionLeftStable(inds, [1, 3, 5, 7])
    assert inds[:4] == approx([1, 3, 5, 7])
    assert inds[4:] == approx([0, 2, 4, 6, 8, 9])

    inds = arange(n)
    partitionRightStable(inds, [1, 3, 5, 7])
    assert inds[-4:] == approx([1, 3, 5, 7])
    assert inds[:-4] == approx([0, 2, 4, 6, 8, 9])
