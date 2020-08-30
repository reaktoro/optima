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
from numpy.testing import assert_array_almost_equal
from pytest import approx, mark
from itertools import product

from utils.matrices import testing_matrices_A


# Tested number of columns
tested_n = [15, 20, 30, 50]

# Tested number of rows
tested_m = range(5, 15, 3)

# Tested cases for the structures of matrix A
tested_matrices_A = testing_matrices_A

# Tested cases for the indices of fixed variables
tested_jfixed = [
    arange(0),
    array([0]),
    array([0, 1, 2])
]

# Combination of all tested cases
testdata = product(
    tested_n,
    tested_m,
    tested_matrices_A,
    tested_jfixed)


def check_canonical_form(canonicalizer, A):
    # Auxiliary varibles
    m, n = A.shape
    R = canonicalizer.R()
    Q = canonicalizer.Q()
    C = canonicalizer.C()

    # Check R*A*Q == C
    Cstar = R @ A[:,Q]

    assert_array_almost_equal(Cstar, C)


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


def check_new_ordering(canonicalizer, Kb, Kn):
    R = array(canonicalizer.R())  # create a copy of internal reference
    S = array(canonicalizer.S())  # create a copy of internal reference
    Q = array(canonicalizer.Q())  # create a copy of internal reference

    canonicalizer.updateOrdering(Kb, Kn)

    Rnew = canonicalizer.R()
    Snew = canonicalizer.S()
    Qnew = canonicalizer.Q()

    nb = len(Kb)

    assert_array_almost_equal(Rnew[:nb, :],  R[Kb, :])
    assert_array_almost_equal(Snew,  S[Kb, :][:, Kn])
    assert_array_almost_equal(Qnew[:nb],  Q[:nb][Kb])
    assert_array_almost_equal(Qnew[nb:],  Q[nb:][Kn])


def check_canonicalizer(canonicalizer, A):
    # Auxiliary variables
    n = canonicalizer.numVariables()
    m = canonicalizer.numEquations()
    nb = canonicalizer.numBasicVariables()
    nn = canonicalizer.numNonBasicVariables()

    #---------------------------------------------------------------------------
    # Check the computed canonical form
    #---------------------------------------------------------------------------
    check_canonical_form(canonicalizer, A)

    #---------------------------------------------------------------------------
    # Perform a series of basis swap operations and check the canonical form
    #---------------------------------------------------------------------------
    for i in range(nb):
        for j in range(0, nn, nb):  # only after every nb columns (to avoid too long test execution)
            S = canonicalizer.S()
            if abs(S[i, j]) > 1e-12:  # new basic variable needs to have sufficiently large pivot value
                canonicalizer.updateWithSwapBasicVariable(i, j)
                canonicalizer.cleanResidualRoundoffErrors()
                check_canonical_form(canonicalizer, A)
        canonicalizer.reset()  # ensure the canonical form is reset periodically so that accumulated round-off errors are removed

    #---------------------------------------------------------------------------
    # Set weights for the variables to update the basic/non-basic partition
    #---------------------------------------------------------------------------
    weigths = random.rand(n)

    canonicalizer.updateWithPriorityWeights(weigths)
    canonicalizer.cleanResidualRoundoffErrors()

    check_canonical_form(canonicalizer, A)

    check_canonical_ordering(canonicalizer, weigths)

    #---------------------------------------------------------------------------
    # Check changing ordering of basic and non-basic variables work
    #---------------------------------------------------------------------------
    Kb = list(range(nb))
    Kn = list(range(nn))

    check_new_ordering(canonicalizer, Kb, Kn)  # identical ordering

    Kb = list(reversed(range(nb)))
    Kn = list(reversed(range(nn)))

    check_new_ordering(canonicalizer, Kb, Kn)  # reversed ordering


@mark.parametrize("args", testdata)
def test_canonicalizer(args):

    n, m, assemble_A, jfixed = args

    A = assemble_A(m, n, jfixed)

    canonicalizer = Canonicalizer(A)
    canonicalizer.cleanResidualRoundoffErrors()
    check_canonicalizer(canonicalizer, A)
