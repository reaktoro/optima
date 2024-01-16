# Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
#
# Copyright © 2020-2024 Allan Leal
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


from testing.optima import *
from testing.utils.matrices import *


# Tested number of columns
tested_n = [10, 15, 20, 30]

# Tested number of rows
tested_m = [5, 10, 15, 20]

# Tested cases for the structures of matrix A
tested_assembleA = testing_matrices_A

# Tested cases for the indices of fixed variables
tested_jfixed = [
    npy.arange(0),
    npy.array([0]),
    npy.array([0, 1, 2])
]


def check_canonical_form(echelonizer, A):
    # Auxiliary varibles
    m, n = A.shape
    R = echelonizer.R()
    Q = echelonizer.Q()
    C = echelonizer.C()

    # Check R*A*Q == C
    Cstar = R @ A[:,Q]

    nb = echelonizer.numBasicVariables()
    nl = m - nb
    Cstar[nb:, :] = 0.0

    assert_array_almost_equal(Cstar, C)


def check_canonical_ordering(echelonizer, weights):
    n = echelonizer.numVariables()
    nb = echelonizer.numBasicVariables()
    nn = echelonizer.numNonBasicVariables()
    ibasic = echelonizer.indicesBasicVariables()
    inonbasic = echelonizer.indicesNonBasicVariables()
    for i in range(1, nb):
        assert weights[ibasic[i]] <= weights[ibasic[i - 1]]
    for i in range(1, nn):
        assert weights[inonbasic[i]] <= weights[inonbasic[i - 1]]


def check_new_ordering(echelonizer, Kb, Kn):
    R = npy.array(echelonizer.R())  # create a copy of internal reference
    S = npy.array(echelonizer.S())  # create a copy of internal reference
    Q = npy.array(echelonizer.Q())  # create a copy of internal reference

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
    weights = npy.linspace(n, 1, n)

    echelonizer.updateWithPriorityWeights(weights)
    echelonizer.cleanResidualRoundoffErrors()

    check_canonical_form(echelonizer, A)

    check_canonical_ordering(echelonizer, weights)

    #---------------------------------------------------------------------------
    # Check changing ordering of basic and non-basic variables work
    #---------------------------------------------------------------------------
    Kb = list(range(nb))
    Kn = list(range(nn))

    check_new_ordering(echelonizer, Kb, Kn)  # identical ordering

    Kb = list(reversed(range(nb)))
    Kn = list(reversed(range(nn)))

    check_new_ordering(echelonizer, Kb, Kn)  # reversed ordering



@pytest.mark.parametrize("n"        , tested_n)
@pytest.mark.parametrize("m"        , tested_m)
@pytest.mark.parametrize("assembleA", tested_assembleA)
@pytest.mark.parametrize("jfixed"   , tested_jfixed)
def testEchelonizer(n, m, assembleA, jfixed):

    A = assembleA(m, n, jfixed)

    echelonizer = Echelonizer(A)
    echelonizer.cleanResidualRoundoffErrors()
    check_echelonizer(echelonizer, A)

    #==============================================================
    # Check if Echelonizer::compute performs memoization correctly
    #==============================================================
    echelonizer = Echelonizer(A)

    weights = npy.linspace(1, n, n)
    echelonizer.updateWithPriorityWeights(weights)

    R = npy.copy(echelonizer.R())
    S = npy.copy(echelonizer.S())
    Q = npy.copy(echelonizer.Q())
    C = npy.copy(echelonizer.C())

    echelonizer.compute(A)  # use same A, then ensure R, S, Q, C remain identical

    Rnew = npy.copy(echelonizer.R())
    Snew = npy.copy(echelonizer.S())
    Qnew = npy.copy(echelonizer.Q())
    Cnew = npy.copy(echelonizer.C())

    assert npy.array_equal(R, Rnew)
    assert npy.array_equal(S, Snew)
    assert npy.array_equal(Q, Qnew)
    assert npy.array_equal(C, Cnew)

    A = npy.ones((m, n))  # change A, then ensure R, Q, C have changed accordingly
    echelonizer.compute(A)

    Rnew = npy.copy(echelonizer.R())
    Snew = npy.copy(echelonizer.S())
    Qnew = npy.copy(echelonizer.Q())

    assert not npy.array_equal(R, Rnew)
