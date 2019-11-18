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
from numpy.linalg import norm, inv
from pytest import approx, mark


def assemble_matrix_A_with_linearly_independent_rows_only(m, n, nf=0):
    A = eigen.randomSPD(n)
    return A[:m, :]


def assemble_matrix_A_with_one_linearly_dependent_row(m, n, nf=0):
    A = assemble_matrix_A_with_linearly_independent_rows_only(m, n, nf)
    A[2, :] = 2*A[0, :] + A[1, :]
    return A


def assemble_matrix_A_with_two_linearly_dependent_rows(m, n, nf=0):
    A = assemble_matrix_A_with_linearly_independent_rows_only(m, n, nf)
    A[2, :] = 2*A[0, :] + A[1, :]
    A[3, :] = A[1, :]
    return A


def assemble_matrix_A_with_one_basic_fixed_variable(m, n, nf=0):
    A = assemble_matrix_A_with_linearly_independent_rows_only(m, n, nf)
    A[-1, -nf] = 0.0
    return A


def assemble_matrix_A_with_two_basic_fixed_variables(m, n, nf=0):
    A = assemble_matrix_A_with_linearly_independent_rows_only(m, n, nf)
    A[-2, -nf] = 0.0
    A[-1, -nf] = 0.0
    return A


tested_matrices_A = [
    assemble_matrix_A_with_linearly_independent_rows_only,
    assemble_matrix_A_with_one_linearly_dependent_row,
    assemble_matrix_A_with_two_linearly_dependent_rows,
    assemble_matrix_A_with_one_basic_fixed_variable,
    assemble_matrix_A_with_two_basic_fixed_variables,
]


testdata = tested_matrices_A


def reverse(list):
    return list[::-1]


def check_canonical_form(canonicalizer, A):
    # Auxiliary varibles
    m, n = A.shape
    R = canonicalizer.R()
    Q = canonicalizer.Q()
    C = canonicalizer.C()

    # Check R*A*Q == C
    assert norm(R.dot(A[:,Q]) - C) / norm(C) == approx(0.0)

    # Assemble Qtr, the transpose of the permutation matrix Q
    Qtr = arange(n)
    Qtr[Q] = arange(n)

    # Calculate the invR, the inverse of matrix R
    Rinv = inv(R)

    # Check inv(R) * C * tr(Q) == A
    assert Rinv.dot(C[:, Qtr]) == approx(A)


def check_canonical_ordering(canonicalizer, weigths):
    n = canonicalizer.numVariables()
    nb = canonicalizer.numBasicVariables()
    nn = canonicalizer.numNonBasicVariables()
    ibasic = canonicalizer.indicesBasicVariables()
    inonbasic = canonicalizer.indicesNonBasicVariables()
    for i in range(1, nb):
        assert weigths[ibasic[i]] <= weigths[ibasic[i - 1]]
    for i in range(1, nn):
        assert weigths[inonbasic[i]] <= weigths[inonbasic[i - 1]]


def check_canonicalizer(canonicalizer, A):
    # Auxiliary variables
    n = canonicalizer.numVariables()
    m = canonicalizer.numEquations()
    nb = canonicalizer.numBasicVariables()

    #---------------------------------------------------------------------------
    # Check the computed canonical form
    #---------------------------------------------------------------------------
    check_canonical_form(canonicalizer, A)

    #---------------------------------------------------------------------------
    # Perform a series of basis swap operations and check the canonical form
    #---------------------------------------------------------------------------
    for i in range(nb):
        for j in range(n - nb):
            canonicalizer.updateWithSwapBasicVariable(i, j)
            check_canonical_form(canonicalizer, A)

    #---------------------------------------------------------------------------
    # Set weights for the variables to update the basic/non-basic partition
    #---------------------------------------------------------------------------
    weigths = abs(random.rand(n)) + 1.0

    canonicalizer.updateWithPriorityWeights(weigths)

    check_canonical_form(canonicalizer, A)

    check_canonical_ordering(canonicalizer, weigths)


@mark.parametrize("assemble_A", tested_matrices_A)
def test_canonicalizer(assemble_A):
    m = 4
    n = 6

    A = assemble_A(m, n)

    canonicalizer = Canonicalizer(A)
    check_canonicalizer(canonicalizer, A)
