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
from numpy.linalg import norm
from numpy.testing import assert_array_almost_equal
from pytest import approx, mark
from itertools import product

from utils.matrices import testing_matrices_A


# Tested number of columns
tested_n = [15, 20]

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


def check_canonical_form(echelonizer, A):
    # Auxiliary varibles
    m, n = A.shape
    R = echelonizer.R()
    Q = echelonizer.Q()
    C = echelonizer.C()

    # Check R*A*Q == C
    Cstar = R @ A[:,Q]

    assert_array_almost_equal(Cstar, C)


def check_canonical_ordering(echelonizer, weigths):
    n = echelonizer.numVariables()
    nb = echelonizer.numBasicVariables()
    nn = echelonizer.numNonBasicVariables()
    ibasic = echelonizer.indicesBasicVariables()
    inonbasic = echelonizer.indicesNonBasicVariables()
    for i in range(1, nb):
        assert weigths[ibasic[i]] <= weigths[ibasic[i - 1]]
    for i in range(1, nn):
        assert weigths[inonbasic[i]] <= weigths[inonbasic[i - 1]]


def check_new_ordering(echelonizer, Kb, Kn):
    R = array(echelonizer.R())  # create a copy of internal reference
    S = array(echelonizer.S())  # create a copy of internal reference
    Q = array(echelonizer.Q())  # create a copy of internal reference

    echelonizer.updateOrdering(Kb, Kn)

    Rnew = echelonizer.R()
    Snew = echelonizer.S()
    Qnew = echelonizer.Q()

    nb = len(Kb)

    assert_array_almost_equal(Rnew[:nb, :],  R[Kb, :])
    assert_array_almost_equal(Snew,  S[Kb, :][:, Kn])
    assert_array_almost_equal(Qnew[:nb],  Q[:nb][Kb])
    assert_array_almost_equal(Qnew[nb:],  Q[nb:][Kn])


def check_echelonizer(echelonizer, A):
    # Auxiliary variables
    n = echelonizer.numVariables()
    m = echelonizer.numEquations()
    nb = echelonizer.numBasicVariables()
    nn = echelonizer.numNonBasicVariables()

    #---------------------------------------------------------------------------
    # Check the computed canonical form
    #---------------------------------------------------------------------------
    check_canonical_form(echelonizer, A)

    #---------------------------------------------------------------------------
    # Perform a series of basis swap operations and check the canonical form
    #---------------------------------------------------------------------------
    for i in range(nb):
        for j in range(0, nn, nb):  # only after every nb columns (to avoid too long test execution)
            S = echelonizer.S()
            if abs(S[i, j]) > 1e-12:  # new basic variable needs to have sufficiently large pivot value
                echelonizer.updateWithSwapBasicVariable(i, j)
                echelonizer.cleanResidualRoundoffErrors()
                check_canonical_form(echelonizer, A)
        echelonizer.reset()  # ensure the canonical form is reset periodically so that accumulated round-off errors are removed

    #---------------------------------------------------------------------------
    # Set weights for the variables to update the basic/non-basic partition
    #---------------------------------------------------------------------------
    weigths = random.rand(n)

    echelonizer.updateWithPriorityWeights(weigths)
    echelonizer.cleanResidualRoundoffErrors()

    check_canonical_form(echelonizer, A)

    check_canonical_ordering(echelonizer, weigths)

    #---------------------------------------------------------------------------
    # Check changing ordering of basic and non-basic variables work
    #---------------------------------------------------------------------------
    Kb = list(range(nb))
    Kn = list(range(nn))

    check_new_ordering(echelonizer, Kb, Kn)  # identical ordering

    Kb = list(reversed(range(nb)))
    Kn = list(reversed(range(nn)))

    check_new_ordering(echelonizer, Kb, Kn)  # reversed ordering


@mark.parametrize("args", testdata)
def testEchelonizer(args):

    n, m, assemble_A, jfixed = args

    A = assemble_A(m, n, jfixed)

    echelonizer = Echelonizer(A)
    echelonizer.cleanResidualRoundoffErrors()
    check_echelonizer(echelonizer, A)

    #==============================================================
    # Check if Echelonizer::compute performs memoization correctly
    #==============================================================
    echelonizer = Echelonizer(A)

    weigths = random.rand(n)
    echelonizer.updateWithPriorityWeights(weigths)

    R = copy(echelonizer.R())
    Q = copy(echelonizer.Q())
    C = copy(echelonizer.C())

    echelonizer.compute(A)  # use same A, then ensure R, Q, C remain identical

    Rnew = copy(echelonizer.R())
    Qnew = copy(echelonizer.Q())
    Cnew = copy(echelonizer.C())

    assert all(R == Rnew)
    assert all(Q == Qnew)
    assert all(C == Cnew)

    A = random.rand(m, n)  # change A, then ensure R, Q, C have changed accordingly
    echelonizer.compute(A)

    Rnew = copy(echelonizer.R())
    Qnew = copy(echelonizer.Q())
    Cnew = copy(echelonizer.C())

    assert not all(R == Rnew)
    assert not all(Q == Qnew)
    assert not all(C == Cnew)
