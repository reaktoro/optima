// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2014-2018 Allan Leal
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
namespace py = pybind11;

// Optima includes
#include <Optima/BasicSolver.hpp>
#include <Optima/Options.hpp>
#include <Optima/Result.hpp>
using namespace Optima;

void exportBasicSolver(py::module& m)
{
    /// The data needed in the constructor of class BasicSolver.
    using BasicSolverInitArgs4py = std::tuple<
        Index,            ///< n       The number of primal variables *x*.
        Index,            ///< m       The number of linear and nonlinear equality constraints.
        MatrixConstRef,   ///< A       The coefficient matrix *A* of the linear equality constraints.
        IndicesConstRef,  ///< ilower  The indices of the variables with lower bounds.
        IndicesConstRef,  ///< iupper  The indices of the variables with upper bounds.
        IndicesConstRef   ///< ifixed  The indices of the variables with fixed values.
    >;

    /// The data needed in method BasicSolver::solve.
    using BasicSolverSolveArgs4py = std::tuple<
        ObjectiveFunction const&,  ///< obj     The objective function *f(x)* of the basic optimization problem.
        ConstraintFunction const&, ///< h       The nonlinear equality constraint function *h(x)*.
        VectorConstRef,            ///< b       The right-hand side vector *b* of the linear equality constraints *Ax = b*.
        VectorConstRef,            ///< xlower  The lower bound values of the variables with lower bounds.
        VectorConstRef,            ///< xupper  The upper bound values of the variables with upper bounds.
        VectorConstRef             ///< xfixed  The values of the variables with fixed values.
    >;

    /// The calculated data from method BasicSolver::solve.
    using BasicSolverSolution4py = std::tuple<
        VectorRef,      ///< x       The calculated primal variables *x* of the basic optimization problem.
        VectorRef,      ///< y       The calculated Lagrange multipliers *y* with respect to constraints *Ax = b* and *h(x) = 0*.
        VectorRef,      ///< z       The *instability measures* of the primal variables defined as *z = g + tr(A)yl + tr(J)yn*.
        IndexNumberRef, ///< nul     The number of lower unstable variables (i.e. those active/attached at their lower bound)
        IndexNumberRef, ///< nuu     The number of upper unstable variables (i.e. those active/attached at their upper bound)
        IndicesRef,     ///< ilower  The indices of the variables with lower bounds organized so that the first `nul` are unstable.
        IndicesRef      ///< iupper  The indices of the variables with upper bounds organized so that the first `nuu` are unstable.
    >;

    auto init = [](const BasicSolverInitArgs4py& args) -> BasicSolver
    {
        const auto [n, m, A, ilower, iupper, ifixed] = args;
        return BasicSolver({n, m, A, ilower, iupper, ifixed});
    };

    auto solve = [](BasicSolver& self, const BasicSolverSolveArgs4py& args, BasicSolverSolution4py sol) -> Result
    {
        const auto [obj, h, b, xlower, xupper, xfixed] = args;
        auto [x, y, z, nul, nuu, ilower, iupper] = sol;
        return self.solve({obj, h, b, xlower, xupper, xfixed}, {x, y, z, nul, nuu, ilower, iupper});
    };

    py::class_<BasicSolver>(m, "BasicSolver")
        .def(py::init<>())
        .def(py::init(init))
        .def("setOptions", &BasicSolver::setOptions)
        .def("solve", solve)
        ;
}
