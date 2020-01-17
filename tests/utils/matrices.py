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


def matrix_with_linearly_independent_rows_only(m, n, nfixed=0):
    """Return a matrix with linearly independent rows only.

    Arguments:
        m {int} -- The number of rows in the matrix
        n {int} -- The number of columns in the matrix

    Keyword Arguments:
        nfixed {int} -- The number of fixed (default: {0})

    Returns:
        [type] -- [description]
    """
    A = eigen.randomSPD(n)
    return A[:m, :]


def matrix_with_one_linearly_dependent_row(m, n, nfixed=0):
    A = matrix_with_linearly_independent_rows_only(m, n, nfixed)
    A[2, :] = 2*A[0, :] + A[1, :]
    return A


def matrix_with_two_linearly_dependent_rows(m, n, nfixed=0):
    A = matrix_with_linearly_independent_rows_only(m, n, nfixed)
    A[2, :] = 2*A[0, :] + A[1, :]
    A[3, :] = A[1, :]
    return A


def matrix_with_one_basic_fixed_variable(m, n, nfixed=0):
    A = matrix_with_linearly_independent_rows_only(m, n, nfixed)
    A[-1, -nfixed] = 0.0
    return A


def matrix_with_two_basic_fixed_variables(m, n, nfixed=0):
    A = matrix_with_linearly_independent_rows_only(m, n, nfixed)
    A[-2, -nfixed] = 0.0
    A[-1, -nfixed] = 0.0
    return A


tested_matrices_A = [
    matrix_with_linearly_independent_rows_only,
    matrix_with_one_linearly_dependent_row,
    matrix_with_two_linearly_dependent_rows,
    matrix_with_one_basic_fixed_variable,
    matrix_with_two_basic_fixed_variables,
]


testdata = tested_matrices_A

