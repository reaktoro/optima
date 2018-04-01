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

def test_partition():
    n = 10
    partition = Partition(n)
    
    partition.setFixedVariables([4, 5, 6])
    assert set(partition.fixedVariables()) == set([4, 5, 6])
    assert set(partition.freeVariables()) == set(arange(n)) - set([4, 5, 6])
    assert partition.numVariables() == n
    assert partition.numFixedVariables() == 3
    assert partition.numFreeVariables() == n - 3
    assert set(partition.ordering()) == set(arange(n))
    
    partition.setFreeVariables([4, 5, 6])
    assert set(partition.freeVariables()) == set([4, 5, 6])
    assert set(partition.fixedVariables()) == set(arange(n)) - set([4, 5, 6])
    assert partition.numVariables() == n
    assert partition.numFixedVariables() == n - 3
    assert partition.numFreeVariables() == 3
    assert set(partition.ordering()) == set(arange(n))
