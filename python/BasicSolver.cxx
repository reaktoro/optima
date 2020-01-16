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
namespace py = pybind11;

// Optima includes
#include <Optima/BasicSolver.hpp>
#include <Optima/Options.hpp>
#include <Optima/Result.hpp>
using namespace Optima;

/// An workaround to expose ObjectiveFunction to python because of ObjectiveResult reference argument.
using ObjectiveFunction4Py = std::function<void(VectorConstRef, ObjectiveResult*)>;

/// An workaround to expose BasicProblem to python because of ObjectiveFunction.
struct BasicProblem4Py
{
    /// The dimensions of the basic optimization problem.
    BasicDims dims;

    /// The constraints of the basic optimization problem.
    BasicConstraints constraints;

    /// The objective function of the basic optimization problem.
    ObjectiveFunction4Py objective;
};

/// Convert an ObjectiveFunction4Py function to an ObjectiveFunction function.
auto convert(const ObjectiveFunction4Py& obj4py)
{
    return [=](VectorConstRef x, ObjectiveResult& f) { obj4py(x, &f); };
}

/// Convert a BasicProblem4Py objecto to a BasicProblem object.
auto convert(const BasicProblem4Py& problem4py)
{
    return BasicProblem{
        problem4py.dims,
        problem4py.constraints,
        convert(problem4py.objective)
    };
}

void exportBasicSolver(py::module& m)
{
    py::class_<BasicState>(m, "BasicState")
        .def_readwrite("x", &BasicState::x, "The primal variables of the basic optimization problem.")
        .def_readwrite("y", &BasicState::y, "The Lagrange multipliers with respect to the equality constraints Ax = b and h(x) = 0.")
        .def_readwrite("z", &BasicState::z, "The slack variables with respect to the lower bounds of the primal variables.")
        .def_readwrite("w", &BasicState::w, "The slack variables with respect to the upper bounds of the primal variables.")
        ;

    py::class_<BasicDims>(m, "BasicDims")
        .def_readwrite("x", &BasicDims::x, "The number of variables (equivalent to the dimension of vector x).")
        .def_readwrite("b", &BasicDims::b, "The number of linear equality constraint equations (equivalent to the dimension of vector b).")
        .def_readwrite("h", &BasicDims::h, "The number of non-linear equality constraint equations (equivalent to the dimension of vector h).")
        .def_readwrite("xlower", &BasicDims::xlower, "The number of variables with lower bounds (equivalent to the dimension of vector of lower bounds xl).")
        .def_readwrite("xupper", &BasicDims::xupper, "The number of variables with upper bounds (equivalent to the dimension of vector of upper bounds xu).")
        .def_readwrite("xfixed", &BasicDims::xfixed, "The number of variables with fixed values (equivalent to the dimension of vector xf).")
        ;

    py::class_<BasicConstraints>(m, "BasicConstraints")
        .def_readwrite("A", &BasicConstraints::A, "The coefficient matrix of the linear equality constraint equations Ax = b.")
        .def_readwrite("h", &BasicConstraints::h, "The constraint function in the non-linear equality constraint equations h(x) = 0.")
        .def_readwrite("ilower", &BasicConstraints::ilower, "The indices of the variables with lower bounds.")
        .def_readwrite("iupper", &BasicConstraints::iupper, "The indices of the variables with upper bounds.")
        .def_readwrite("ifixed", &BasicConstraints::ifixed, "The indices of the variables with fixed values.")
        ;

    py::class_<BasicParams>(m, "BasicParams")
        .def_readwrite("b", &BasicParams::b, "The right-hand side vector of the linear equality constraints Ax = b.")
        .def_readwrite("xlower", &BasicParams::xlower, "The lower bounds of the variables in x that have lower bounds.")
        .def_readwrite("xupper", &BasicParams::xupper, "The upper bounds of the variables x that have upper bounds.")
        .def_readwrite("xfixed", &BasicParams::xfixed, "The values of the variables in x that are fixed.")
        .def_readwrite("extra", &BasicParams::extra, "The extra parameters in the problem.")
        ;

    py::class_<BasicProblem4Py>(m, "BasicProblem")
        .def_readwrite("dims", &BasicProblem4Py::dims, "The dimensions of the basic optimization problem.")
        .def_readwrite("constraints", &BasicProblem4Py::constraints, "The constraints of the basic optimization problem.")
        .def_readwrite("objective", &BasicProblem4Py::objective, "The objective function of the basic optimization problem.")
        ;

	// A constructor function for BasicSolver that accepts BasicProblem4Py instead of BasicProblem.
    auto initBasicSolver = [](const BasicProblem4Py& problem4py)
    {
        return BasicSolver(convert(problem4py));
    };

    py::class_<BasicSolver>(m, "BasicSolver")
        .def(py::init<>())
        .def(py::init(initBasicSolver))
        .def("setOptions", &BasicSolver::setOptions)
        .def("solve", &BasicSolver::solve)
        ;
}
