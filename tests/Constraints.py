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


def test_constraints():
    n = 10
    m = 5

    A = eigen.random(m, n)

    constraints = Constraints(n)
    constraints.setEqualityConstraintMatrix(A)

    constraints.setVariablesWithLowerBounds([0, 2])
    assert constraints.variablesWithLowerBounds() == approx([0, 2])
    assert constraints.variablesWithoutLowerBounds() == approx([1, 3, 4, 5, 6, 7, 8, 9])

    constraints.setVariablesWithUpperBounds([2, 4])
    assert constraints.variablesWithUpperBounds() == approx([2, 4])
    assert constraints.variablesWithoutUpperBounds() == approx([0, 1, 3, 5, 6, 7, 8, 9])

    constraints.setVariablesWithFixedValues([6, 8, 9])
    assert constraints.variablesWithFixedValues() == approx([6, 8, 9])
    assert constraints.variablesWithoutFixedValues() == approx([0, 1, 2, 3, 4, 5, 7])

    constraints.allVariablesHaveLowerBounds()
    assert constraints.variablesWithLowerBounds() == approx(arange(n))

    constraints.allVariablesHaveUpperBounds()
    assert constraints.variablesWithUpperBounds() == approx(arange(n))

    assert constraints.numVariables() == n
    assert constraints.numEqualityConstraints() == m

    # Assert orderingLowerBounds = [without lower bounds, with lower bounds]
    assert list(constraints.orderingLowerBounds()) == \
        list(constraints.variablesWithoutLowerBounds()) + \
             list(constraints.variablesWithLowerBounds())

    # Assert orderingUpperBounds = [without upper bounds, with upper bounds]
    assert list(constraints.orderingUpperBounds()) == \
        list(constraints.variablesWithoutUpperBounds()) + \
             list(constraints.variablesWithUpperBounds())

    # Assert orderingFixedValues = [without fixed values, with fixed values]
    assert list(constraints.orderingFixedValues()) == \
        list(constraints.variablesWithoutFixedValues()) + \
             list(constraints.variablesWithFixedValues())

