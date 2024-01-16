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
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


from testing.optima import *
from testing.utils.matrices import *


# Tested number of variables in x
tested_n = [20, 40, 60]

# Tested rank deficiency of matrix A
tested_rank_deficiency = [0, 1, 5, 10, 15]


@pytest.mark.parametrize("n", tested_n)
@pytest.mark.parametrize("rank_deficiency", tested_rank_deficiency)
def testLU(n, rank_deficiency):


    def check(A, x_expected, rank_expected, linearly_dependent_rows):
        b = A @ x_expected
        lu = LU()
        x = npy.zeros(n)
        lu.decompose(A)
        lu.solve(b, x)

        assert_allclose(A @ x, b)


    x = npy.linspace(1, n, n)

    linearly_dependent_rows = list(range(1, n, math.ceil(n / rank_deficiency))) \
        if rank_deficiency != 0 else []

    rank_expected = n - len(linearly_dependent_rows)

    A = matrix_non_singular(n)

    # Change the rows of A so that linearly dependent rows are produced
    for row in linearly_dependent_rows:
        A[row, :] = row * A[0, :]

    check(A, x, rank_expected, linearly_dependent_rows)
