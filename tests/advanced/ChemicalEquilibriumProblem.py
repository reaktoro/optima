# # Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
# #
# # Copyright © 2020-2024 Allan Leal
# #
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# # GNU General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with this program. If not, see <http://www.gnu.org/licenses/>.

# from optima import *
# from numpy import *
# from numpy.linalg import norm
# from pytest import approx, mark

# from utils.matrices import assemble_matrix_Ax, matrix_non_singular, pascal_matrix

# # The elements in the chemical system
# elements = ["H", "C", "O", "Na", "Mg", "Si", "Cl", "Ca", "Z"]

# # The species in the chemical system
# species = ["H2O", "H+", "OH-", "H2", "O2", "Na+", "Cl-", "NaCl",
#            "HCl", "NaOH", "Ca++", "Mg++", "CH4", "CO2", "HCO3-",
#            "CO3--", "CaCl2", "CaCO3", "MgCO3", "SiO2", "CO2(g)",
#            "O2(g)", "H2(g)", "H2O(g)", "CH4(g)", "CO(g)", "Halite",
#            "Calcite", "Magnesite", "Dolomite", "Quartz"]

# # The formula matrix of the chemical system
# A = array([
#     [2,  1,  1,  2,  0,  0,  0,  0,  1,  1,  0,  0,  4,  0,  1,  0,  0,  0,  0,  0,  0,  0,  2,  2,  4,  0,  0,  0,  0,  0,  0],
#     [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  0,  1,  1,  0,  1,  0,  0,  0,  1,  1,  0,  1,  1,  2,  0],
#     [1,  0,  1,  0,  2,  0,  0,  0,  0,  1,  0,  0,  0,  2,  3,  3,  0,  3,  3,  2,  2,  2,  0,  1,  0,  1,  0,  3,  3,  6,  2],
#     [0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
#     [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0],
#     [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
#     [0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
#     [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0],
#     [0,  1, -1,  0,  0,  1, -1,  0,  0,  0,  2,  2,  0,  0, -1, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
# ], dtype=float)

# # The standard Gibbs energies of the species
# c = array([
#      -237181.72, # 0:  H2O
#            0.00, # 1:  H+
#      -157297.48, # 2:  OH-
#        17723.42, # 3:  H2
#        16543.54, # 4:  O2
#      -261880.74, # 5:  Na+
#      -131289.74, # 6:  Cl-
#      -388735.44, # 7:  NaCl
#      -127235.44, # 8:  HCl
#      -417981.60, # 9:  NaOH
#      -552790.08, # 10: Ca++
#      -453984.92, # 11: Mg++
#       -34451.06, # 12: CH4
#      -385974.00, # 13: CO2
#      -586939.89, # 14: HCO3-
#      -527983.14, # 15: CO3--
#      -811696.00, # 16: CaCl2
#     -1099764.40, # 17: CaCO3
#      -998971.84, # 18: MgCO3
#      -833410.96, # 19: SiO2
#      -394358.74, # 20: CO2(g)
#            0.00, # 21: O2(g)
#            0.00, # 22: H2(g)
#      -228131.76, # 23: H2O(g)
#       -50720.12, # 24: CH4(g)
#      -137168.26, # 25: CO(g)
#      -384120.49, # 26: Halite
#     -1129177.92, # 27: Calcite
#     -1027833.07, # 28: Magnesite
#     -2166307.84, # 29: Dolomite
#      -856238.86  # 30: Quartz
# ])


# def i(speciesname):
#     return species.index(speciesname)

# def j(elementsymbol):
#     return elements.index(elementsymbol)

# # Tested cases for the saddle point methods
# tested_methods = [
#     SaddlePointMethod.Fullspace,
#     # SaddlePointMethod.Nullspace,
#     # SaddlePointMethod.Rangespace,
# ]


# def create_objective_fn():
#     def fn(x, p, res):
#         res.f   = c.T @ x
#         res.fx  = c
#         res.fxx[:] = 0.0
#         res.fxp[:] = 0.0
#     return fn


# def create_constraint_hfn():
#     def fn(x, p, res):
#         pass
#     return fn


# def create_constraint_vfn():
#     def fn(x, p, res):
#         pass
#     return fn

# @mark.parametrize("method", tested_methods)
# def test_basic_solver(method):

#     # Create the objective function with assembled Hxx, Hxp, cx
#     obj = create_objective_fn()

#     # Create the nonlinear equality constraint function h(x, p)
#     h = create_constraint_hfn()

#     # Create the nonlinear external constraint function v(x, p)
#     v = create_constraint_vfn()

#     nx = len(species)
#     np = 0
#     ny = len(elements)
#     nz = 0

#     Ax = A
#     Ap = zeros((0,0))

#     # Create vectors for the solution of the optimization problem
#     x = zeros(nx)
#     p = zeros(np)
#     y = zeros(ny)
#     z = zeros(nz)
#     s = zeros(nx)

#     # Create the stability state of the variables
#     stability = Stability()

#     # Create the options for the optimization calculation
#     options = Options()
#     # options.output.active = False
#     options.output.active = True
#     options.kkt.method = method
#     options.maxiterations = 10
#     options.linesearch.trigger_when_current_error_is_greater_than_initial_error_by_factor = 1.0
#     options.linesearch.trigger_when_current_error_is_greater_than_previous_error_by_factor = 1.0


#     x = zeros(nx)
#     x[i("H2O")]       = 55.0
#     x[i("O2")]        = 1.e-6
#     x[i("CO2(g)")]    = 1.0
#     x[i("Halite")]    = 1.0
#     x[i("Calcite")]   = 1.0
#     x[i("Magnesite")] = 1.0
#     x[i("Dolomite")]  = 1.0
#     x[i("Quartz")]    = 1.0

#     # Compute vector b in Ax*x = b
#     b = A @ x

#     xlower = zeros(nx)
#     xupper = full(nx,  inf)

#     plower = full(np, -inf)
#     pupper = full(np,  inf)

#     # Solve the optimization problem
#     solver = BasicSolver(nx, np, ny, nz, Ax, Ap)
#     solver.setOptions(options)

#     res = solver.solve(obj, h, v, b, xlower, xupper, plower, pupper, x, p, y, z, s, stability)

#     if not res.succeeded:

#         # set_printoptions(linewidth=100000, formatter={'float': '{: 0.3f}'.format})
#         set_printoptions(linewidth=100000, precision=6, suppress=True)
#         print()
#         # print(f"H = \n{H}\n")
#         # print(f"A = \n{A}\n")
#         print(f"x(actual)   = {x}")
#         print(f"x(expected) = {x_expected}")
#         print(f"x(diff) = {abs(x - x_expected)}")
#         print(f"p(actual)   = {p}")
#         print(f"p(expected) = {p_expected}")
#         print(f"p(diff) = {abs(p - p_expected)}")
#         print(f"y(actual)   = {y}")
#         print(f"y(expected) = {y_expected}")
#         print(f"y(diff) = {abs(y - y_expected)}")
#         print(f"z(actual)   = {z}")
#         print(f"z(expected) = {z_expected}")
#         print(f"z(diff) = {abs(z - z_expected)}")

#     assert res.succeeded
