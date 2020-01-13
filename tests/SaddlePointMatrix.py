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
from itertools import product


# The number of variables
n = 15

# Tested cases for the structure of matrix H
tested_structures_H = ['dense', 'diagonal']

# Tested cases for the structure of matrix D
tested_structures_D = ['diagonal', 'zero']

# Tested cases for the structure of matrix G
tested_structures_G = ['dense', 'zero']

# Tested cases for the indices of fixed variables
tested_jf = [arange(0),
             arange(1),
             array([1, 3, 7, 9])]

# Tested number of rows in matrix Au and Al (upper and lower blocks of A)
tested_mu = [7, 1, 0]
tested_ml = [5, 1, 0]

# Combination of all tested cases
testdata = product(tested_structures_H,
                   tested_structures_D,
                   tested_structures_G,
                   tested_jf,
                   tested_mu,
                   tested_ml)


@mark.parametrize("args", testdata)
def test_saddle_point_matrix(args):

    structure_H, structure_D, structure_G, jf, mu, ml = args

    m = mu + ml  # the number of rows of matrix A = [Au; Al]
    t = n + m

    # Create matrices H, A = [Au; Al], and G
    H = eigen.random(n, n) if structure_H == 'dense' else eigen.random(n)
    D = eigen.random(n) if structure_D == 'diagonal' else eigen.vector()
    G = eigen.random(m, m) if structure_G == 'dense' else eigen.matrix()
    A = eigen.random(m, n)

    # The upper and lower blocks of matrix A
    Au = A[:mu, :]
    Al = A[mu:, :]

    # Create the SaddlePointMatrix object
    mat = SaddlePointMatrix(H, D, Au, Al, G, jf)

    # Use a dense matrix for H from this point on (for convenience)
    Haux = H if structure_H == 'dense' else eigen.diag(H)

    # Use a dense matrix for D from this point on (for convenience)
    Daux = eigen.diag(D) if structure_D == 'diagonal' else eigen.zeros(n, n)

    # Use a dense matrix for G from this point on (for convenience)
    Gaux = G if structure_G == 'dense' else eigen.zeros(m ,m)

    # Assemble the tr(A) matrix block, with zeros on rows corresponding to fixed variables
    trA = transpose(A.copy())
    trA[jf, :] = 0.0

    # Set to zero the rows and columns of H corresponding to fixed variables
    Haux[jf, :] = 0.0
    Haux[:, jf] = 0.0

    # Set to one the diagonal entries in H corresponding to fixed variables
    Haux[jf, jf] = 1.0

    # Set to zero the diagonal entries in D corresponding to fixed variables
    Daux[jf, jf] = 0.0

    # Assemble the big saddle point matrix M
    M = eigen.zeros(t, t)
    M[:n, :n] = Haux + Daux
    M[:n, n:n + m] = trA
    M[n:, :n] = A
    M[n:, n:] = Gaux

    assert mat.array() == approx(M)


def test_saddle_point_vector():
    m = 5
    n = 15
    t = n + m

    r = arange(1, t + 1)

    vec = SaddlePointVector(r, n, m)

    assert vec.array() == approx(r)
