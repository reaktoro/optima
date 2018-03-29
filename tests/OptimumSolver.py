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

def objective(x):
    res = ObjectiveResult()
    res.f = sum((x - 1)**2)
    res.g = 2.0 * (x - 1)
    res.H = 2.0 * ones(len(x))
    return res

def test_optimum_solver():
    n = 2
    m = 1

    structure = OptimumStructure(n, m)
    structure.allVariablesHaveLowerBounds()
    structure.allVariablesHaveUpperBounds()
    structure.A = [[1.0, -1.0]]

    params = OptimumParams()
    params.b = [0.0]
    params.xlower = [0.0, 0.0]
    params.xupper = [5.0, 5.0] 
    params.objective = objective
    
    state = OptimumState()
    state.x = array([2.0, 2.0])

    options = OptimumOptions()
    options.output.active = True
    
    solver = OptimumSolver(structure)
    solver.setOptions(options)
    solver.solve(params, state)
    
    print state.x

