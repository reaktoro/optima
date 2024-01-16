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


def check_canonical_form(echelonizer, A, J):
    # Auxiliary varibles
    M = npy.concatenate([A, J], 0)
    m, n = M.shape
    R = echelonizer.R()
    Q = echelonizer.Q()
    C = echelonizer.C()

    # Check R*[A; J]*Q == C
    Cstar = R @ M[:,Q]

    nb = echelonizer.numBasicVariables()
    nl = m - nb
    Cstar[nb:, :] = 0.0

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


def check_echelonizer(echelonizer, A, J):
    # Auxiliary variables
    n = echelonizer.numVariables()
    m = echelonizer.numEquations()

    #---------------------------------------------------------------------------
    # Check the computed canonical form with certain weights
    #---------------------------------------------------------------------------
    weigths = npy.linspace(1, n, n)

    echelonizer.updateWithPriorityWeights(J, weigths)
    echelonizer.cleanResidualRoundoffErrors()

    check_canonical_form(echelonizer, A, J)

    #---------------------------------------------------------------------------
    # Set weights for the variables to update the basic/non-basic partition
    #---------------------------------------------------------------------------
    weigths = npy.linspace(n, 1, n)

    Jnew = J + J

    echelonizer.updateWithPriorityWeights(Jnew, weigths)
    echelonizer.cleanResidualRoundoffErrors()

    check_canonical_form(echelonizer, A, Jnew)

    check_canonical_ordering(echelonizer, weigths)

    #---------------------------------------------------------------------------
    # Check changing ordering of basic and non-basic variables work
    #---------------------------------------------------------------------------
    nb = echelonizer.numBasicVariables()
    nn = echelonizer.numNonBasicVariables()

    Kb = list(range(nb))
    Kn = list(range(nn))

    check_new_ordering(echelonizer, Kb, Kn)  # identical ordering

    Kb = list(reversed(range(nb)))
    Kn = list(reversed(range(nn)))

    check_new_ordering(echelonizer, Kb, Kn)  # reversed ordering


# Tested number of columns in W = [A; J]
tested_nx = [10, 15, 20, 30]

# Tested number of rows in A
tested_ny = [5, 10, 15, 20]

# Tested number of rows in J
tested_nz = [0, 5, 10]

# Tested cases for the matrix W = [A; J]
tested_assembleA = [
    matrix_with_linearly_independent_rows_only,
    # matrix_with_one_linearly_dependent_row,
    # matrix_with_two_linearly_dependent_rows,  # Currently, this setup fails. See note in EchelonizerExtended.cpp::updateWithPriorityWeights.
    matrix_with_one_basic_fixed_variable,
    matrix_with_two_basic_fixed_variables,
    matrix_with_one_zero_column,
    matrix_with_two_zero_columns
]


@pytest.mark.parametrize("nx"       , tested_nx)
@pytest.mark.parametrize("ny"       , tested_ny)
@pytest.mark.parametrize("nz"       , tested_nz)
@pytest.mark.parametrize("assembleA", tested_assembleA)
def testEchelonizerExtended(nx, ny, nz, assembleA):

    nw = ny + nz

    # Skip tests in which there are more rows than columns
    if nw > nx: return

    if nz >= ny: pytest.xfail("Currently, when nz >= ny, not very accurate results are produced.")
    if nz >= 10: pytest.xfail("Currently, when nz >= 10, not very accurate results are produced.")

    A = assembleA(ny, nx)
    J = pascal_matrix(nz, nx)

    #----------------------------------------------------------------------------------------------
    # WARNING!!
    #
    # EchelonizerExtended assumes that rows in J are not linearly dependent
    # on rows in A. When testing, ensure this does not happen. This assumption
    # should be sensible in most application cases. However, when testing, it
    # may happen that W is produced so that this assumption is not respected.
    #----------------------------------------------------------------------------------------------

    echelonizer = EchelonizerExtended(A)
    echelonizer.cleanResidualRoundoffErrors()
    check_echelonizer(echelonizer, A, J)
