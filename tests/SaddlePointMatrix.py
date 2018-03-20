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

testdata = [(zeroG, numfixed)
            for zeroG in [True, False] 
            for numfixed in [0, 5]]


@mark.parametrize("args", testdata)
def test_saddle_point_matrix_dense(args):
    m = 5
    n = 15
    
    zeroG, nf = args
    
    nx = n - nf
    
    A = eigen.ones(m, n)
    H = eigen.ones(n, n)
    G = eigen.matrix() if zeroG else eigen.ones(m, m)
    
    M = eigen.zeros(n + m, n + m)
    M[0:nx, 0:nx] = H[:nx, :nx]
    M[nx:n, nx:n] = eye(nf, nf)
    M[0:nx, n:n+m] = transpose(A[:, 0:nx])
    M[n:, :n] = A
    if not zeroG: M[n:, n:] = G

    mat = SaddlePointMatrix(H, A, G, nf)
    
    assert mat.array() == approx(M)
    
 
@mark.parametrize("args", testdata)
def test_saddle_point_matrix_diagonal(args):
    m = 5
    n = 15
 
    zeroG, nf = args
         
    nx = n - nf
     
    A = eigen.random(m, n)
    H = eigen.random(n)
    G = eigen.matrix() if zeroG else eigen.random(m, m)
     
    M = eigen.zeros(n + m, n + m)
    M[0:nx, 0:nx] = eigen.diag(H[:nx])
    M[nx:n, nx:n] = eye(nf, nf)
    M[0:nx, n:n+m] = transpose(A[:, 0:nx])
    M[n:, :n] = A
    if not zeroG: M[n:, n:] = G
 
    mat = SaddlePointMatrix(H, A, G, nf)
     
    assert mat.array() == approx(M)


def test_saddle_point_vector():
    m = 5
    n = 15
    t = n + m

    r = arange(1, t + 1)

    vec = SaddlePointVector(r, n, m)

    assert vec.array() == approx(r)
