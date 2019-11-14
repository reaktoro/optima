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
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from optima import *
from numpy import *
from pytest import approx, mark


def test_optimum_structure():
    n = 10
    m = 5

    structure = Structure(n, m)
    structure.A = eigen.random(m, n)

    structure.setVariablesWithLowerBounds([0, 2])
    assert structure.variablesWithLowerBounds() == approx([0, 2])
    assert structure.variablesWithoutLowerBounds() == approx([1, 3, 4, 5, 6, 7, 8, 9])

    structure.setVariablesWithUpperBounds([2, 4])
    assert structure.variablesWithUpperBounds() == approx([2, 4])
    assert structure.variablesWithoutUpperBounds() == approx([0, 1, 3, 5, 6, 7, 8, 9])

    structure.setVariablesWithFixedValues([6, 8, 9])
    assert structure.variablesWithFixedValues() == approx([6, 8, 9])
    assert structure.variablesWithoutFixedValues() == approx([0, 1, 2, 3, 4, 5, 7])

    structure.allVariablesHaveLowerBounds()
    assert structure.variablesWithLowerBounds() == approx(arange(n))

    structure.allVariablesHaveUpperBounds()
    assert structure.variablesWithUpperBounds() == approx(arange(n))

    assert structure.numVariables() == n
    assert structure.numEqualityConstraints() == m

    # Assert orderingLowerBounds = [without lower bounds, with lower bounds]
    assert list(structure.orderingLowerBounds()) == \
        list(structure.variablesWithoutLowerBounds()) + \
             list(structure.variablesWithLowerBounds())

    # Assert orderingUpperBounds = [without upper bounds, with upper bounds]
    assert list(structure.orderingUpperBounds()) == \
        list(structure.variablesWithoutUpperBounds()) + \
             list(structure.variablesWithUpperBounds())

    # Assert orderingFixedValues = [without fixed values, with fixed values]
    assert list(structure.orderingFixedValues()) == \
        list(structure.variablesWithoutFixedValues()) + \
             list(structure.variablesWithFixedValues())

