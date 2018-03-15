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
from pytest import approx


def test_ip_saddle_point_matrix():
    H = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    A = array([[1, 2, 3], [3, 4, 5]])
    Z = array([1, 2, 3])
    W = array([4, 5, 6])
    L = array([9, 8, 7])
    U = array([6, 5, 4])

    n = 3
    m = 2

    # Testing conversion when no variables are fixed
    nx = 3
    nf = 0

    mat = IpSaddlePointMatrix(H, A, Z, W, L, U, nf)

    M = [[1,  2,  3,  1,  3, -1,  0,  0, -1,  0,  0],
         [4,  5,  6,  2,  4,  0, -1,  0,  0, -1,  0],
         [7,  8,  9,  3,  5,  0,  0, -1,  0,  0, -1],
         [1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0],
         [3,  4,  5,  0,  0,  0,  0,  0,  0,  0,  0],
         [1,  0,  0,  0,  0,  9,  0,  0,  0,  0,  0],
         [0,  2,  0,  0,  0,  0,  8,  0,  0,  0,  0],
         [0,  0,  3,  0,  0,  0,  0,  7,  0,  0,  0],
         [4,  0,  0,  0,  0,  0,  0,  0,  6,  0,  0],
         [0,  5,  0,  0,  0,  0,  0,  0,  0,  5,  0],
         [0,  0,  6,  0,  0,  0,  0,  0,  0,  0,  4]]
    
    M = array(M, dtype=float64)

    # Check conversion to a Matrix instance
    assert mat.array() == approx(M)

    # Testing conversion when some variables are fixed
    nx = 2
    nf = 1
 
    mat = IpSaddlePointMatrix(H, A, Z, W, L, U, nf)
 
    M = [[1,  2,  0,  1,  3, -1,  0,  0, -1,  0,  0],
         [4,  5,  0,  2,  4,  0, -1,  0,  0, -1,  0],
         [0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
         [1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0],
         [3,  4,  5,  0,  0,  0,  0,  0,  0,  0,  0],
         [1,  0,  0,  0,  0,  9,  0,  0,  0,  0,  0],
         [0,  2,  0,  0,  0,  0,  8,  0,  0,  0,  0],
         [0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
         [4,  0,  0,  0,  0,  0,  0,  0,  6,  0,  0],
         [0,  5,  0,  0,  0,  0,  0,  0,  0,  5,  0],
         [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]]
 
    M = array(M, dtype=float64)
 
    # Check conversion to a Matrix instance
    assert mat.array() == approx(M)


def test_ip_saddle_point_vector():
    n = 5
    m = 3
    t = 3*n + m
 
    r = linspace(1, t, t)
 
    vec = IpSaddlePointVector(r, n, m)
 
    assert r == approx(concatenate((vec.a, vec.b, vec.c, vec.d)))
