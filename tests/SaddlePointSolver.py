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

def test_saddle_point_solver():
    m = 10
    n = 60
    t = m + n

    expected = linspace(1, t, t);

    A = eigen.random(m, n)

    def check_saddle_point_solver_with_dense_H(nf=0, zeroG=False):

        H = eigen.random(n, n)
        G = eigen.random(m, m) if not zeroG else eigen.matrix()
        nx = n - nf;
        
        lhs = SaddlePointMatrix(H, A, G, nx, nf)
        
        r = lhs.matrix().dot(expected)
        s = zeros(t)

        rhs = SaddlePointVector(r, n, m)
        sol = SaddlePointSolution(s, n, m)

        methods = [
            SaddlePointMethod.FullPivLU,
            SaddlePointMethod.PartialPivLU,
            SaddlePointMethod.Nullspace,
            ]
        
        for method in methods:
            options = SaddlePointOptions()
            options.method = method
            
            solver = SaddlePointSolver()
            solver.setOptions(options)
            solver.initialize(lhs.A())
            solver.decompose(lhs)
            solver.solve(rhs, sol)
            assert expected == approx(s)
            
    def check_saddle_point_solver_with_diagonal_H(nf=0, zeroG=False):

        H = eigen.diag(eigen.random(n))  
        G = eigen.random(m, m) if not zeroG else eigen.matrix()
        nx = n - nf;
        
        lhs = SaddlePointMatrix(H, A, G, nx, nf)
        
        r = lhs.matrix().dot(expected)
        s = zeros(t)

        rhs = SaddlePointVector(r, n, m)
        sol = SaddlePointSolution(s, n, m)

        options = SaddlePointOptions()
        options.method = SaddlePointMethod.Rangespace
        
        solver = SaddlePointSolver()
        solver.setOptions(options)
        solver.initialize(lhs.A())
        solver.decompose(lhs)
        solver.solve(rhs, sol)
        assert expected == approx(s)
            

    nf = 0; zeroG = False  # No fixed variables, G is dense 
    
    check_saddle_point_solver_with_dense_H(nf, zeroG)
    check_saddle_point_solver_with_diagonal_H(nf, zeroG)

    nf = 0; zeroG = True  # No fixed variables, G is zero matrix
    
    check_saddle_point_solver_with_dense_H(nf, zeroG)
    check_saddle_point_solver_with_diagonal_H(nf, zeroG)
    
    nf = n / 10; zeroG = False  # With fixed variables, G is dense

    check_saddle_point_solver_with_dense_H(nf, zeroG)
    check_saddle_point_solver_with_diagonal_H(nf, zeroG)

    nf = n / 10; zeroG = True  # With fixed variables, G is zero matrix

    check_saddle_point_solver_with_dense_H(nf, zeroG)
    check_saddle_point_solver_with_diagonal_H(nf, zeroG)

