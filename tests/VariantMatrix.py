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


def test_variant_matrix():
    n = 10
       
    mat = eigen.ones(n, n)
    
    vmat = VariantMatrix(mat)    
    
    assert vmat.structure == MatrixStructure.Dense
    assert vmat.dense == approx(mat)
    
    vec = eigen.ones(n)
    
    vmat = VariantMatrix(vec)
    
    assert vmat.structure == MatrixStructure.Diagonal
    assert vmat.diagonal == approx(vec)
    
    vmat = VariantMatrix()
    
    vmat.setZero() 
    assert vmat.structure == MatrixStructure.Zero
    
    vmat.setDiagonal(n); 
    assert vmat.structure == MatrixStructure.Diagonal
    assert len(vmat.diagonal) == n
    
    vmat.setDense(n); 
    assert vmat.structure == MatrixStructure.Dense
    assert vmat.dense.shape == (n, n)


def test_variant_matrix_const_ref():
    n = 10
    
    mat = eigen.ones(n, n)
    
    vmat = VariantMatrixConstRef(mat)
    
    assert vmat.structure == MatrixStructure.Dense
    assert vmat.dense == approx(mat)
    
    vec = eigen.ones(n)
    
    vmat = VariantMatrixConstRef(vec)
    
    assert vmat.structure == MatrixStructure.Diagonal
    assert vmat.diagonal == approx(vec)

