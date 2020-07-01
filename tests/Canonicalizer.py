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
from itertools import product

from utils.matrices import testing_matrices_W


# Tested cases for the structures of matrix W = [A; J]
tested_matrices_W = testing_matrices_W

# Tested cases for the indices of fixed variables
tested_jfixed = [
    arange(0),
    array([0]),
    array([0, 1, 2])
]

# Combination of all tested cases
testdata = product(
    tested_matrices_W,
    tested_jfixed)


def check_canonical_form(canonicalizer, A):
    # Auxiliary varibles
    m, n = A.shape
    R = canonicalizer.R()
    Q = canonicalizer.Q()
    C = canonicalizer.C()


    # Check R*A*Q == C
    Cstar = R @ A[:,Q]

    assert allclose(Cstar, C)

    # Assemble Qtr, the transpose of the permutation matrix Q
    Qtr = arange(n)
    Qtr[Q] = arange(n)

    # Calculate the invR, the inverse of matrix R
    Rinv = inv(R)

    # Check inv(R) * C * tr(Q) == A
    Astar = Rinv @ C[:, Qtr]

    assert allclose(Astar, A)


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
            S = canonicalizer.S()
            if abs(S[i, j]) > 1e-12:  # new basic variable needs to have sufficiently large pivot value
                canonicalizer.updateWithSwapBasicVariable(i, j)
                canonicalizer.cleanResidualRoundoffErrors()
                check_canonical_form(canonicalizer, A)

    #---------------------------------------------------------------------------
    # Set weights for the variables to update the basic/non-basic partition
    #---------------------------------------------------------------------------
    weigths = random.rand(n)

    canonicalizer.updateWithPriorityWeights(weigths)
    canonicalizer.cleanResidualRoundoffErrors()

    check_canonical_form(canonicalizer, A)

    check_canonical_ordering(canonicalizer, weigths)


@mark.parametrize("args", testdata)
def test_canonicalizer(args):

    assemble_W, jfixed = args

    m = 12
    n = 15

    A = assemble_W(m, n, jfixed)

    canonicalizer = Canonicalizer(A)
    canonicalizer.cleanResidualRoundoffErrors()
    check_canonicalizer(canonicalizer, A)
