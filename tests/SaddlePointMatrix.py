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

structure_options_H = ['dense', 'diagonal']
structure_options_G = ['dense', 'zero']
num_fixed_variables_options = [0, 5]
 
testdata = [(x, y, z) 
            for x in structure_options_H
            for y in structure_options_G
            for z in num_fixed_variables_options] 

@mark.parametrize("args", testdata)
def test_saddle_point_matrix(args):
    m = 5
    n = 15
    
    structureH, structureG, nf = args
    
    nx = n - nf
    
    A = eigen.random(m, n)
    
    M = eigen.zeros(n + m, n + m)
    
    if structureH == 'dense':
        H = eigen.random(n, n)
        M[0:nx, 0:nx] = H[:nx, :nx]
    else: # diagonal
        H = eigen.random(n)
        M[0:nx, 0:nx] = eigen.diag(H[:nx])
    
    if structureG == 'dense':
        G = eigen.random(m, m)
        M[n:, n:] = G
    else: # zero
        G = eigen.matrix()
    
    M[nx:n, nx:n] = eye(nf, nf)
    M[0:nx, n:n+m] = transpose(A[:, 0:nx])
    M[n:, :n] = A

    # Create the SaddlePointMatrix object
    print 'H:', H
    mat = SaddlePointMatrix(H, A, G, nf)

    print 'structure:', mat.H.structure
    print 'H.dense\n', mat.H.dense
    print 'H.diagonal\n', mat.H.diagonal
    print 'H:', H
    
    # Check conversion to a Matrix instance
    assert mat.array() == approx(M)


def test_saddle_point_vector():
    n = 5
    m = 3
    t = n + m

    r = arange(1, t + 1)

    vec = SaddlePointVector(r, n, m)

    assert vec.array() == approx(r)
