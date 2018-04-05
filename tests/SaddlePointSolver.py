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

import Canonicalizer

# Tested cases for the matrix A
tested_matrices_A = Canonicalizer.tested_matrices_A

# Tested cases for the structure of matrix H
tested_structures_H = ['dense', 'diagonal']

# Tested cases for the structure of matrix G
tested_structures_G = ['dense', 'zero']

# Tested cases for the indices of fixed variables
tested_jf = [arange(0), 
             arange(1), 
             array([1, 3, 7, 9])]

# Tested cases for the conditions of the variables in terms of pivot variables
tested_variable_conditions = ['all-variables-pivot',
                              'all-variables-nonpivot',
                              'some-variables-pivot']

# Tested cases for the saddle point methods
tested_methods = [
    SaddlePointMethod.Fullspace,
    SaddlePointMethod.Nullspace,
    SaddlePointMethod.Rangespace,
    ]

# Combination of all tested cases
testdata = product(tested_matrices_A,
                   tested_structures_H,
                   tested_structures_G,
                   tested_jf,
                   tested_variable_conditions,
                   tested_methods)


@mark.parametrize("args", testdata)
def test_saddle_point_solver(args):

    assemble_A, structure_H, structure_G, jf, variable_condition, method = args

    m = 4
    n = 10

    t = m + n

    expected = linspace(1, t, t);

    A = assemble_A(m, n, len(jf))

    H = eigen.random(n, n) if structure_H == 'dense' else eigen.random(n)
    G = eigen.random(m, m) if structure_G == 'dense' else eigen.matrix()

    # The diagonal entries of the Hessian matrix
    Hdiag = H[diag_indices(n)] if structure_H == 'dense' else H

    # The sequence along the diagonal that is affected to control the number of pivot variables
    seq = slice(m) if variable_condition == 'some-variables-pivot' else slice(n)  

    # The factor multiplied by the entries in the diagonal of the Hessian matrix
    factor = 1e-6 if variable_condition == 'all-variables-nonpivot' else 1e6
    
    # Adjust the diagonal entries to control number of pivot variables
    Hdiag[seq] = factor * Hdiag[seq] 

    # Create the SaddlePointMatrix object
    lhs = SaddlePointMatrix(H, A, G, jf)
    
    # Use the SaddlePointMatrix object to create an array M
    M = lhs.array()
    
    # Compute the right-hand side vector r = M * expected
    r = M.dot(expected)
    
    # The solution vector
    s = zeros(t)

    # The right-hand side and solution saddle point vectors
    rhs = SaddlePointVector(r, n, m)
    sol = SaddlePointSolution(s, n, m)

    # Specify the saddle point method for the current test
    options = SaddlePointOptions()
    options.method = method

    # Create a SaddlePointSolver to solve the saddle point problem
    solver = SaddlePointSolver()
    solver.setOptions(options)
    solver.initialize(lhs.A)
    solver.decompose(lhs)
    solver.solve(rhs, sol)

    # Check the residual of the equation M * s = r
    assert norm(M.dot(s) - r) / norm(r) == approx(0.0)
