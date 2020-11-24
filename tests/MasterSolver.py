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

from numpy import *


# def testMasterSolverSimpleSines():

#     def objectivefn_f(res, x, p, opts):
#         res.f   = -sum(sin(x))
#         res.fx  = -cos(x)
#         res.fxx = diag(sin(x))
#         res.fxp[:] = 0.0
#         res.diagfxx = False
#         res.fxx4basicvars = False
#         res.succeeded = True

#     def constraintfn_h(res, x, p, opts):
#         pass

#     def constraintfn_v(res, x, p, opts):
#         pass

#     problem = MasterProblem()
#     problem.f = objectivefn_f
#     problem.h = constraintfn_h
#     problem.v = constraintfn_v
#     problem.Ax = array([[1.0, -1.0]])
#     problem.Ap = zeros((0, 0))
#     problem.b = array([0.0])
#     problem.xlower = array([0.0, 0.0])
#     problem.xupper = array([pi, pi])
#     problem.phi = None

#     options = Options()

#     nx, np, ny, nz = 2, 0, 1, 0

#     dims = MasterDims(nx, np, ny, nz)

#     solver = MasterSolver(dims)

#     solver.setOptions(options)

#     u = MasterVector(dims)
#     u.x = [1.0, 1.0]

#     res = solver.solve(problem, u)

#     assert res.succeeded




# def testMasterSolverAdvancedSines():

#     def objectivefn_f(res, x, p, opts):
#         nx = len(x)
#         np = len(p)
#         aux = sum(p * sin(x))
#         res.f   = -0.5*aux*aux
#         # res.fx  = -aux*p*cos(x)
#         for i in range(nx):
#             res.fx[i] = -aux*p[i]*cos(x[i])
#             for j in range(nx):
#                 res.fxx[i, j] = -p[i] * p[j] * cos(x[i]) * cos(x[j])
#             # for j in range(np):
#             #     res.fxp[i, j] = -p[i] * cos(x[i]) * sin(x[j])
#             res.fxx[i, i] += aux * p[i] * sin(x[i])
#             # res.fxp[i, i] -= aux * cos(x[i])
#         # print()
#         # print(f"f = {res.f}")
#         # print(f"g = {res.fx}")
#         # print(f"x = {x}")
#         # print(f"p = {p}")
#         w, v = linalg.eig(res.fxx)
#         # print(f"eigvals(Hxx) = {w}")
#         print(f"eigvals(Hxx-before) = {linalg.eigvals(res.fxx)}")
#         # w = array([max(x, 0) for x in w])
#         # w = abs(w)
#         # res.fxx = v @ diag(w) @ linalg.inv(v)
#         # print(f"eigvals(Hxx-after)  = {linalg.eigvals(res.fxx)}")
#         # print(f"g = \n{res.fx}")  #1234
#         # print(f"Hxx = \n{res.fxx}")  #1234
#         # print(f"Hxp = \n{res.fxp}")  #1234
#         res.diagfxx = False
#         res.fxx4basicvars = False
#         res.succeeded = True

#     def constraintfn_h(res, x, p, opts):
#         nx = len(x)
#         nz = len(res.val)
#         m = nx - nz - 1
#         for i in range(m, nx - 1):
#             k = i - m
#             aux = exp(x[i + 1] - x[i])
#             res.val[k] = aux - 1.0
#             res.ddx[k, i] = -aux
#             res.ddx[k, i + 1] = aux
#         # print(f"Jx = \n{res.ddx}")  #1234
#         # print(f"Jp = \n{res.ddp}")  #1234
#         res.ddx4basicvars = False
#         res.succeeded = True

#     def constraintfn_v(res, x, p, opts):
#         nx = len(x)
#         np = len(p)
#         # xb = sum(x) / nx
#         # aux = exp(2*xb*p/pi)
#         # for i in range(nx):
#         #     res.val[i] = aux[i] - 1
#         #     for j in range(nx):
#         #         res.ddx[i, j] = aux[i] * (2 * p[j])/(nx * pi)
#         #     res.ddp[i, i] = aux[i] * (2 * xb)/pi
#         res.val = p - 1
#         res.ddp = eye(np)
#         # aux = exp(p - 1)
#         # res.val = aux - 1
#         # res.ddp = diag(aux)
#         # print(f"Vx = \n{res.ddx}")  #1234
#         # print(f"Vp = \n{res.ddp}")  #1234

#         # res.val = res.ddp.T @ res.val  # Mimic minimization of 1/2||v(x, p)||2
#         # res.ddx = res.ddp.T @ res.ddx  # Mimic minimization of 1/2||v(x, p)||2
#         # res.ddp = res.ddp.T @ res.ddp  # Mimic minimization of 1/2||v(x, p)||2

#         res.ddx4basicvars = False
#         res.succeeded = True


#     nx = np = 4

#     m = int(nx/2)

#     ny = m
#     nz = nx - m - 1

#     Ax = zeros((m, nx))
#     for i in range(m):
#         Ax[i, i] = -1.0
#         Ax[i, i + 1] = 1.0

#     # print(f"Ax = \n{Ax}")  #1234
#     Ap = zeros((m, np))

#     problem = MasterProblem()
#     problem.f = objectivefn_f
#     problem.h = constraintfn_h
#     problem.v = constraintfn_v
#     problem.Ax = copy(Ax)
#     problem.Ap = copy(Ap)
#     problem.b = zeros(m)
#     # problem.xlower = full(nx, pi/2 - pi/4)
#     # problem.xupper = full(nx, pi/2 + pi/4)
#     # problem.xlower = full(nx, pi/2 - 3.7)
#     # problem.xupper = full(nx, pi/2 + 3.7)
#     problem.xlower = full(nx, 0.1)
#     problem.xupper = full(nx, 3.14)
#     problem.plower = full(np, 0.0 + 0.1)
#     problem.pupper = full(np, 2.0 - 0.1)
#     # problem.plower = full(np, -nx*pi)
#     # problem.pupper = full(np, -nx*pi + pi)
#     problem.phi = None

#     options = Options()
#     options.maxiterations = 100
#     options.newtonstep.linearsolver.method = LinearSolverMethod.Nullspace

#     dims = MasterDims(nx, np, ny, nz)

#     solver = MasterSolver(dims)

#     solver.setOptions(options)

#     u = MasterVector(dims)
#     # u.x = ones(nx)
#     u.x = zeros(nx)
#     u.p = zeros(nx)

#     res = solver.solve(problem, u)

#     print(f"u.x = \n{u.x}")
#     print(f"u.p = \n{u.p}")
#     print(f"res.iterations = \n{res.iterations}")

#     assert res.succeeded







# def testMasterSolverAdvancedSines():

#     def objectivefn_f(res, x, p, opts):
#         nx = len(x)
#         np = len(p)
#         xE = sum(x)
#         aux1 = sum(sin(xE + p))
#         aux2 = sum(cos(xE + p))
#         res.f = -aux1
#         for i in range(nx):
#             res.fx[i] = -aux2
#             for j in range(nx):
#                 res.fxx[i, j] = aux1
#             for j in range(np):
#                 res.fxp[i, j] = sin(xE + p[j])
#         print(f"x = {x}")
#         print(f"p = {p}")
#         print()
#         print(f"g = \n{res.fx}")  #123
#         print(f"Hxx = \n{res.fxx}")  #123
#         print(f"Hxp = \n{res.fxp}")  #123
#         res.diagfxx = False
#         res.fxx4basicvars = False
#         res.succeeded = True

#     def constraintfn_h(res, x, p, opts):
#         nx = len(x)
#         nz = len(res.val)
#         m = nx - nz - 1
#         for i in range(m, nx - 1):
#             k = i - m
#             aux = exp(x[i + 1] - x[i])
#             res.val[k] = aux - 1.0
#             res.ddx[k, i] = -aux
#             res.ddx[k, i + 1] = aux
#         print(f"Jx = \n{res.ddx}")  #123
#         print(f"Jp = \n{res.ddp}")  #123
#         res.ddx4basicvars = False
#         res.succeeded = True

#     def constraintfn_v(res, x, p, opts):
#         nx = len(x)
#         xE = sum(x)
#         for i in range(nx):
#             res.val[i] = xE + p[i] - x[i]
#             res.ddx[i, :] = 1.0
#             res.ddx[i, i] = 0.0
#             res.ddp[i, i] = 1.0
#         print(f"Vx = \n{res.ddx}")  #123
#         print(f"Vp = \n{res.ddp}")  #123

#         res.val = res.ddp.T @ res.val  # Mimic minimization of 1/2||v(x, p)||2
#         res.ddx = res.ddp.T @ res.ddx  # Mimic minimization of 1/2||v(x, p)||2
#         res.ddp = res.ddp.T @ res.ddp  # Mimic minimization of 1/2||v(x, p)||2

#         res.ddx4basicvars = False
#         res.succeeded = True


#     nx = np = 10

#     m = int(nx/2)

#     ny = m
#     nz = nx - m - 1

#     Ax = zeros((m, nx))
#     for i in range(m):
#         Ax[i, i] = -1.0
#         Ax[i, i + 1] = 1.0

#     print(f"Ax = \n{Ax}")  #123
#     Ap = zeros((m, np))

#     problem = MasterProblem()
#     problem.f = objectivefn_f
#     problem.h = constraintfn_h
#     problem.v = constraintfn_v
#     problem.Ax = copy(Ax)
#     problem.Ap = copy(Ap)
#     problem.b = zeros(m)
#     problem.xlower = zeros(nx)
#     problem.xupper = full(nx, pi)
#     problem.plower = full(np, -15)
#     problem.pupper = full(np, -10)
#     # problem.plower = full(np, -nx*pi)
#     # problem.pupper = full(np, -nx*pi + pi)
#     problem.phi = None

#     options = Options()
#     options.maxiterations = 10
#     options.newtonstep.linearsolver.method = LinearSolverMethod.Nullspace

#     dims = MasterDims(nx, np, ny, nz)

#     solver = MasterSolver(dims)

#     solver.setOptions(options)

#     u = MasterVector(dims)
#     u.x =  ones(nx)
#     u.p = -ones(np) * (nx - 1)
#     u.p = full(np, -11)
#     # u.p =  ones(np) * 0.1
#     # u.p =  -ones(np) * 0.9
#     # u.x =   random.rand(nx)
#     # u.p =  -random.rand(np)

#     res = solver.solve(problem, u)

#     print(f"u.x = \n{u.x}")
#     print(f"u.p = \n{u.p}")
#     print(f"res.iterations = \n{res.iterations}")

#     assert res.succeeded






























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


# tested_nx      = [10]          # The tested number of x variables
# tested_np      = [0, 5]        # The tested number of p variables
# tested_ny      = [5]           # The tested number of y variables
# tested_nz      = [0, 2]        # The tested number of z variables
# tested_nl      = [0, 1]        # The tested number of linearly dependent rows in Ax
# tested_nu      = [0, 2]        # The tested number of unstable variables
# tested_diagHxx = [True, False] # The tested diagonal structure of Hxx matrix

tested_nx      = [5]           # The tested number of x variables
tested_np      = [0]           # The tested number of p variables
tested_ny      = [2]           # The tested number of y variables
tested_nz      = [0]           # The tested number of z variables
tested_nl      = [0]           # The tested number of linearly dependent rows in Ax
tested_nu      = [0]           # The tested number of unstable variables
tested_diagHxx = [False] # The tested diagonal structure of Hxx matrix

@pytest.mark.parametrize("nx"     , tested_nx)
@pytest.mark.parametrize("np"     , tested_np)
@pytest.mark.parametrize("ny"     , tested_ny)
@pytest.mark.parametrize("nz"     , tested_nz)
@pytest.mark.parametrize("nl"     , tested_nl)
@pytest.mark.parametrize("nu"     , tested_nu)
@pytest.mark.parametrize("diagHxx", tested_diagHxx)
def testMasterSolver(nx, np, ny, nz, nl, nu, diagHxx):

    # Augment x variables with free slack variables to ensure
    # feasibility of linear and non-linear constraints
    nw = ny + nz
    nx = nx + nw

    Hxx = random.rand(nx, nx)
    Hxp = random.rand(nx, np)
    Vpx = random.rand(np, nx)
    Vpp = random.rand(np, np)
    Ax  = random.rand(ny, nx)
    Ap  = random.rand(ny, np)
    Jx  = random.rand(nz, nx)
    Jp  = random.rand(nz, np)

    # Hxx = block([[random.rand(nx-nw, nx-nw)], [zeros(())]])
    Hxx[:, -nw:] = 0.0
    Hxx[-nw:, :] = 0.0
    Hxx[-nw:, -nw:] = eye(nw)

    Wx = block([random.rand(nw, nx-nw), eye(nw)])  # last columns of Wx are to ensure feasibility of linear and non-linear constraints
    Ax, Jx = vsplit(Wx, [ny])

    set_printoptions(linewidth=1000)
    # print()
    # print(f"Ax = \n{Ax}")
    # print(f"Jx = \n{Jx}")
    # print(f"Hxx = \n{Hxx}")

    Hxx = Hxx.T @ Hxx  # ensure Hxx is positive semi-definite or definite

    c = concatenate([ones(nx-nw), zeros(nw)])
    # print(f"c = \n{c}")

    g = -Hxx @ c

    dims = MasterDims(nx, np, ny, nz)

    # ju  = M.ju
    # js  = M.js

    # dims = params.dims

    cx = random.rand(dims.nx)
    cp = random.rand(dims.np)
    cz = random.rand(dims.nz)

    # cx[ju] = +1.0e4

    def objectivefn_f(res, x, p, opts):
        # res.f   = 0.5 * (x.T @ Hxx @ x) + x.T @ Hxp @ p + g.T @ x + 0.5 * (c.T @ Hxx @ c)
        # res.fx  = Hxx @ x + Hxp @ p + g
        res.f   = 0.5 * (x.T @ Hxx @ x) + x.T @ Hxp @ p
        res.fx  = Hxx @ x + Hxp @ p
        res.fxx = Hxx
        # res.fxp = Hxp
        # print(f"res.f = {res.f}")
        # print(f"x = {x}")
        res.diagfxx = diagHxx
        res.fxx4basicvars = False
        res.succeeded = True

    def constraintfn_h(res, x, p, opts):
        res.val = Jx @ x + Jp @ p + cz
        res.ddx = Jx
        res.ddp = Jp
        res.ddx4basicvars = False
        res.succeeded = True

    def constraintfn_v(res, x, p, opts):
        res.val = Vpx @ x + Vpp @ p + cp
        res.ddx = Vpx
        res.ddp = Vpp
        res.ddx4basicvars = False
        res.succeeded = True

    xlower = npy.full(dims.nx, -npy.inf)
    xupper = npy.full(dims.nx,  npy.inf)

    xlower[0:3] = 2.0
    # xlower[0:1] = 2.0

    # xx = npy.ones(nx)
    # xx[0:3] = 2.0

    px = npy.ones(np)

    # b = Ax @ xx + Ap @ px

    problem = MasterProblem()
    problem.f = objectivefn_f
    problem.h = constraintfn_h
    problem.v = constraintfn_v
    problem.Ax = Ax
    problem.Ap = Ap
    problem.b = random.rand(ny)
    # problem.xlower = npy.full(dims.nx, -npy.inf)
    # problem.xupper = npy.full(dims.nx,  npy.inf)
    problem.xlower = xlower
    problem.xupper = xupper
    problem.phi = None

    options = Options()
    options.maxiterations = 100
    # options.output.active = True

    solver = MasterSolver(dims)

    solver.setOptions(options)

    u = MasterVector(dims)

    res = solver.solve(problem, u)

    assert res.succeeded


