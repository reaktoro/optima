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


from testing.optima import *
from testing.utils.matrices import *

def testMasterSolver():

    def objectivefn_f(res, x, p, opts):
        res.f   = -npy.sum(npy.sin(x))
        res.fx  = -npy.cos(x)
        res.fxx = npy.diag(npy.sin(x))
        res.fxp[:] = 0.0
        res.diagfxx = False
        res.fxx4basicvars = False
        res.succeeded = True

    def constraintfn_h(res, x, p, opts):
        pass

    def constraintfn_v(res, x, p, opts):
        pass

    problem = MasterProblem()
    problem.f = objectivefn_f
    problem.h = constraintfn_h
    problem.v = constraintfn_v
    problem.Ax = npy.array([[1.0, -1.0]])
    problem.Ap = npy.zeros((0, 0))
    problem.b = npy.array([0.0])
    problem.xlower = npy.array([0.0, 0.0])
    problem.xupper = npy.array([npy.pi, npy.pi])
    problem.phi = None

    options = Options()

    nx, np, ny, nz = 2, 0, 1, 0

    dims = MasterDims(nx, np, ny, nz)

    solver = MasterSolver(dims)

    solver.setOptions(options)

    u = MasterVector(dims)
    u.x = [1.0, 1.0]

    res = solver.solve(problem, u)

    assert res.succeeded






























# # Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
# #
# # Copyright (C) 2014-2018 Allan Leal
# #
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# # GNU General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with this program. If not, see <http://www.gnu.org/licenses/>.


# from testing.optima import *
# from testing.utils.matrices import *


# tested_nx = [10]    # The tested number of x variables
# tested_np = [0, 5]  # The tested number of p variables
# tested_ny = [5]     # The tested number of y variables
# tested_nz = [0, 2]  # The tested number of z variables
# tested_nl = [0, 1]  # The tested number of linearly dependent rows in Ax
# tested_nu = [0, 2]  # The tested number of unstable variables


# @pytest.mark.parametrize("nx", tested_nx)
# @pytest.mark.parametrize("np", tested_np)
# @pytest.mark.parametrize("ny", tested_ny)
# @pytest.mark.parametrize("nz", tested_nz)
# @pytest.mark.parametrize("nl", tested_nl)
# @pytest.mark.parametrize("nu", tested_nu)
# def testMasterSolver(nx, np, ny, nz, nl, nu):

#     params = MasterParams(nx, np, ny, nz, nl, nu)

#     if params.invalid(): return

#     M = createMasterMatrix(params)

#     Hxx = M.H.Hxx
#     Hxp = M.H.Hxp
#     Vpx = M.V.Vpx
#     Vpp = M.V.Vpp
#     Ax  = M.W.Ax
#     Ap  = M.W.Ap
#     Jx  = M.W.Jx
#     Jp  = M.W.Jp
#     ju  = M.ju
#     js  = M.js

#     dims = params.dims

#     cx = random.rand(dims.nx)
#     cp = random.rand(dims.np)
#     cz = random.rand(dims.nz)

#     cx[ju] = +1.0e4

#     def objectivefn_f(res, x, p, opts):
#         res.f   = 0.5 * (x.T @ Hxx @ x) + x.T @ Hxp @ p + cx.T @ x
#         res.fx  = Hxx @ x + Hxp @ p + cx
#         res.fxx = Hxx
#         res.fxp = Hxp
#         res.diagfxx = False
#         res.fxx4basicvars = False
#         res.succeeded = True

#     def constraintfn_h(res, x, p, opts):
#         res.val = Jx @ x + Jp @ p + cz
#         res.ddx = Jx
#         res.ddp = Jp
#         res.ddx4basicvars = False
#         res.succeeded = True

#     def constraintfn_v(res, x, p, opts):
#         res.val = Vpx @ x + Vpp @ p + cp
#         res.ddx = Vpx
#         res.ddp = Vpp
#         res.ddx4basicvars = False
#         res.succeeded = True

#     problem = MasterProblem()
#     problem.f = objectivefn_f
#     problem.h = constraintfn_h
#     problem.v = constraintfn_v
#     problem.Ax = Ax
#     problem.Ap = Ap
#     problem.b = random.rand(dims.ny)
#     problem.xlower = -abs(random.rand(dims.nx))
#     problem.xupper =  abs(random.rand(dims.nx))
#     problem.phi = None

#     options = Options()

#     solver = MasterSolver(dims)

#     solver.setOptions(options)

#     u = MasterVector(dims)

#     res = solver.solve(problem, u)

#     assert res.succeeded


