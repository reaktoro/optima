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

    A = eigen.random(m, n)

    structure = OptimumStructure(A)

    structure.setVariablesWithLowerBounds([0, 2])
    assert structure.variablesWithLowerBounds() == approx([0, 2])

    structure.setVariablesWithUpperBounds([2, 4])
    assert structure.variablesWithUpperBounds() == approx([2, 4])

    structure.setVariablesWithFixedValues([6, 8, 9])
    assert structure.variablesWithFixedValues() == approx([6, 8, 9])

    structure.allVariablesHaveLowerBounds()
    assert structure.variablesWithLowerBounds() == approx(arange(n))

    structure.allVariablesHaveUpperBounds()
    assert structure.variablesWithUpperBounds() == approx(arange(n))

    structure.setHessianMatrixAsDense()
    assert structure.structureHessianMatrix() == MatrixStructure.Dense

    structure.setHessianMatrixAsDiagonal()
    assert structure.structureHessianMatrix() == MatrixStructure.Diagonal

    structure.setHessianMatrixAsZero()
    assert structure.structureHessianMatrix() == MatrixStructure.Zero

    assert structure.numVariables() == n
    assert structure.numEqualityConstraints() == m
    assert structure.equalityConstraintMatrix() == approx(A)

