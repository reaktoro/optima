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
from IPython.parallel.controller.scheduler import numpy

# Tested cases for the structure of matrix H
tested_structures_H = ['dense', 'diagonal']

# Tested cases for the indices of fixed variables
tested_jf = [arange(0), 
            arange(1), 
            array([1, 3, 7, 9])
             ]

# Combination of all tested cases
testdata = product(tested_structures_H,
                   tested_jf)


@mark.parametrize("args", testdata)
def test_ip_saddle_point_matrix(args):

    structure_H, jf = args

    m = 5
    n = 15
    t = 3*n + m

    # Create matrices H, A, L, U, Z, W
    H = eigen.random(n, n) if structure_H == 'dense' else eigen.random(n)
    A = eigen.random(m, n)
    L = eigen.random(n)
    U = eigen.random(n)
    Z = eigen.random(n)
    W = eigen.random(n)

    # Create the IpSaddlePointMatrix object
    mat = IpSaddlePointMatrix(H, A, Z, W, L, U, jf).array()

    # Use a dense matrix for H from this point on (for convenience)
    H = H if structure_H == 'dense' else eigen.diag(H)

    # Use a dense matrix for L, U, Z, W from this point on (for convenience)
    L = eigen.diag(L)
    U = eigen.diag(U)
    Z = eigen.diag(Z)
    W = eigen.diag(W)
    
    # Assemble the tr(A) matrix block, with zeros on rows corresponding to fixed variables
    trA = transpose(A.copy())
    trA[jf, :] = 0.0

    # Set to zero the rows and columns of H corresponding to fixed variables
    H[jf, :] = 0.0
    H[:, jf] = 0.0

    # Set to one the diagonal entries in H corresponding to fixed variables
    H[jf, jf] = 1.0

    # Set to one the diagonal entries in L and U corresponding to fixed variables
    L[jf, jf] = U[jf, jf] = 1.0

    # Set to zero the diagonal entries in Z and W corresponding to fixed variables
    Z[jf, jf] = W[jf, jf] = 0.0
    
    # Assemble an identity matrix of dimension n by n
    Inn = eigen.eye(n)
    
    # Set to zero the entries in Inn corresponding to fixed variables
    Inn[jf, jf] = 0.0

    # Assemble zero matrices of dimensions n by m and n by n
    Onm = eigen.zeros(n, m)
    Onn = eigen.zeros(n, n)
    
    # Assemble the big saddle point matrix M 
    M = eigen.zeros(t, t)
    M[0:n, :] = concatenate([H, trA, -Inn, -Inn], 1)
    M[n:n + m, 0:n] = A
    M[n + m:n + m + n, :] = concatenate([Z, Onm, L, Onn], 1)
    M[n + m + n:, :] = concatenate([W, Onm, Onn, U], 1)
    
    assert mat == approx(M)


def test_ip_saddle_point_vector():
    m = 5
    n = 15
    t = 3*n + m

    r = arange(1, t + 1)

    vec = IpSaddlePointVector(r, n, m)

    assert vec.array() == approx(r)
