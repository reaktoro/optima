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
from pytest import approx

def print_state(M, r, s, m, n):
    set_printoptions(linewidth=1000, precision=10)
    slu = linalg.solve(M, r)
    print 'M = \n', M
    print 'r        = ', r
    print 'x        = ', s[:n]
    print 'x(lu)    = ', slu[:n]
    print 'y        = ', s[n:n+m]
    print 'y(lu)    = ', slu[n:n+m]
    print 'z        = ', s[n+m:n+m+n]
    print 'z(lu)    = ', slu[n+m:n+m+n]
    print 'w        = ', s[:n]
    print 'w(lu)    = ', slu[:n]
    print 'res      = ', M.dot(s) - r
    print 'res(lu)  = ', M.dot(slu) - r


def test_ip_saddle_point_solver():
    n = 3
    m = 1
    t = 3*n + m
    nx = n
    nf = 0

    expected = linspace(1, t, t)

#     A = eigen.random(m, n)
#     H = eigen.random(n, n)
#     Z = eigen.random(n)
#     W = eigen.random(n)
#     L = eigen.random(n)
#     U = eigen.random(n)

    A = eigen.ones(m, n)
    H = eigen.ones(n, n)
    Z = eigen.ones(n)
    W = eigen.ones(n)
    L = eigen.ones(n)
    U = eigen.ones(n)

    options = SaddlePointOptions()

    M = eigen.zeros(t, t)
    r = eigen.zeros(t)

    def check_ip_saddle_point_solver():
        # The left-hand side coefficient matrix
        lhs = IpSaddlePointMatrix(H, A, Z, W, L, U, nx, nf)

        # The dense matrix assembled from lhs
        M = lhs.matrix()

        # The right-hand side vector
        r = M.dot(expected)
        rhs = IpSaddlePointVector(r, n, m)

        # The solution vector
        s = eigen.zeros(t)
        sol = IpSaddlePointSolution(s, n, m)

        # Solve the interior-poin saddle point problem
        solver = IpSaddlePointSolver()
        solver.setOptions(options)
        solver.initialize(A)
        solver.decompose(lhs)
        solver.solve(rhs, sol)

        print_state(M, r, s, m, n)

        # Check the residual of the equation Ms = r
        assert norm(M.dot(s) - r)/norm(r) == approx(0.0)

    # When all variables are free
    nx = n
    nf = 0
       
    check_ip_saddle_point_solver()
     
    # When m variables are fixed
    nf = 1
    nx = n - nf
 
    check_ip_saddle_point_solver()

    # When some entries in L are very small
    nx = n
    nf = 0
    L[1] = 1e-16
    Z[1] = 1.0
    W[1] = 1e-16

    check_ip_saddle_point_solver()

