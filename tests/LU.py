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
from numpy.testing import assert_allclose
from pytest import approx, mark
from itertools import product

from utils.matrices import matrix_non_singular

# The number of variables
n = 20

# Combination of all tested cases
testdata = product()

@mark.parametrize("args", testdata)
def test_lu(args):

    _ = args


    def check(A, x_expected, rank_expected):
        b = A @ x_expected
        lu = LU(A)
        x = zeros(n)
        lu.solve(b, x)
        r = lu.rank()

        assert r == rank_expected
        assert norm(A[:r, :r] @ x[:r] - b[:r]) / norm(b)
        assert all(isnan(x[r:]))


    x = linspace(1, n, n)

    r = n
    A = matrix_non_singular(n)

    check(A, x, r)

    r = n - 1
    A = matrix_non_singular(n)
    A[n - r:, :] = 10 * A[:r, :]  # last r rows = first r rows times 10

    check(A, x, r)

    r = n - 2
    A = matrix_non_singular(n)
    A[n - r:, :] = 100 * A[:r, :]  # last r rows = first r rows times 10

    check(A, x, r)
