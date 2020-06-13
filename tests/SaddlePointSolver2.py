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
from numpy.linalg import norm
from pytest import approx, mark
from itertools import product

from utils.matrices import testing_matrices_W, matrix_non_singular


def print_state(M, r, s, m, n):
    set_printoptions(linewidth=1000, suppress=True)
    slu = eigen.solve(M, r)
    print( 'M        = \n', M )
    print( 'r        = ', r )
    print( 'x        = ', s[:n] )
    print( 'x(lu)    = ', slu[:n] )
    print( 'x(diff)  = ', abs(s[:n] - slu[:n]) )
    print( 'y        = ', s[n:n + m] )
    print( 'y(lu)    = ', slu[n:n + m] )
    print( 'y(diff)  = ', abs(s[n:n + m] - slu[n:n + m]) )
    print( 'res      = ', M.dot(s) - r )
    print( 'res(lu)  = ', M.dot(slu) - r )


# The number of variables
n = 20

# Tested cases for the matrix W = [A; J]
tested_matrices_W = testing_matrices_W

# Tested cases for the structure of matrix H
tested_structures_H = [
    'denseH',
    'diagonalH'
]

# Tested cases for the structure of matrix G
tested_structures_G = [
    # 'denseG', # TODO: currently, dense G tests produces more residual error than the other cases (I think it is because of R*G*tr(R) terms, Allan, 21.01.20).
    'zeroG'
]

# Tested cases for the indices of fixed variables
tested_ifixed = [
    arange(0),
    arange(1),
    array([1, 3, 7, 9])
]

# Tested number of rows in matrix A (upper block of W)
tested_mu = [6, 4]
tested_ml = [3, 1, 0]

# Tested cases for the conditions of the variables in terms of pivot variables
tested_variable_conditions = [
    'all-variables-pivot',
    'all-variables-nonpivot',
    'some-variables-pivot'
]

# Tested cases for the saddle point methods
tested_methods = [
    SaddlePointMethod.Fullspace,
    SaddlePointMethod.Nullspace,
    SaddlePointMethod.Rangespace
]

# Combination of all tested cases
testdata = product(tested_matrices_W,
                   tested_structures_H,
                   tested_structures_G,
                   tested_ifixed,
                   tested_mu,
                   tested_ml,
                   tested_variable_conditions,
                   tested_methods)

@mark.parametrize("args", testdata)
def test_saddle_point_solver_2(args):

    assemble_W, structure_H, structure_G, ifixed, ml, mn, variable_condition, method = args

    m = ml + mn

    t = m + n

    nf = len(ifixed)

    # The column-scaling vector D of the Hessian matrix H
    D = ones(n)  # TODO: Parameterize the scaling vector D in the tests of SaddlePointSolver

    # The column-scaling vector V of the coefficient matrix W
    V = ones(n)  # TODO: Parameterize the scaling vector V in the tests of SaddlePointSolver

    # The indices of the rows in H that have zero off-diagonal entries
    idiagonal = []  # TODO: Parameterize the diagonal structure of the Hessian matrix.

    expected = linspace(1, t, t)

    W = assemble_W(m, n, ifixed)

    A = W[:ml, :]  # extract the upper block of W = [A; J]
    J = W[ml:, :]  # extract the lower block of W = [A; J]

    H =  matrix_non_singular(n)
    G = -matrix_non_singular(m) if structure_G == 'denseG' else zeros((m, m))

    if method == SaddlePointMethod.Rangespace:
        H = abs(diag(linspace(1, n, num=n)))

    # The diagonal entries of the Hessian matrix
    Hdiag = H[diag_indices(n)]

    # The sequence along the diagonal that is affected to control the number of pivot variables
    seq = slice(m) if variable_condition == 'some-variables-pivot' else slice(n)

    # The factor multiplied by the entries in the diagonal of the Hessian matrix
    factor = 1e-6 if variable_condition == 'all-variables-nonpivot' else 1e6

    # Adjust the diagonal entries to control number of pivot variables
    Hdiag[seq] = factor * Hdiag[seq]

    # Scale the columns of the Hessian matrix
    HD = H @ diag(D)

    # Scale the columns of the coefficient matrix W
    WV = W @ diag(V)

    # Assemble the coefficient matrix [[H, tr(W)], [W, G]]
    M = block([[HD, WV.T], [WV, G]])

    M[ifixed, :] = 0.0       # zero out rows in M corresponding to fixed variables
    M[:n, ifixed] = 0.0      # zero out cols in M in the top-left block corresponding to fixed variables
    M[ifixed, ifixed] = 1.0  # set to one the diagonal entries in M corresponding to fixed variables

    # Compute the right-hand side vector r = M * expected
    r = M @ expected

    # The right-hand side vectors a and b
    a = r[:n]
    b = r[n:]

    # The solution vectors x and y
    x = a.copy()
    y = b.copy()

    # Set G to empty in case it is zero
    if structure_G == 'zeroG':
        G = zeros((0, 0))

    # Specify the saddle point method for the current test
    options = SaddlePointOptions()
    options.method = method

    # Create a SaddlePointSolver to solve the saddle point problem
    solver = SaddlePointSolver2(n, m, A)
    solver.setOptions(options)
    solver.decompose(H, J, G, D, V, ifixed, idiagonal)
    solver.solve(x, y)

    # Create solution vector s = [x, y]
    s = concatenate([x, y])

    # Check the residual of the equation M * s = r

    tol = 1e-9 if structure_G == 'denseG' else 1e-13

    succeeded = norm(M @ s - r) / norm(r) < tol

    if not succeeded:
        print()
        print(f"assemble_W = {assemble_W}")
        print(f"structure_H = {structure_H}")
        print(f"structure_G = {structure_G}")
        print(f"ifixed = {ifixed}")
        print(f"ml = {ml}")
        print(f"mn = {mn}")
        print(f"variable_condition = {variable_condition}")
        print(f"method = {method}")
        print()

        print_state(M, r, s, m, n)

    assert norm(M @ s - r) / norm(r) < tol