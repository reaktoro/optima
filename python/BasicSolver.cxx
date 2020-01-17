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

/// A workaround to expose BasicConstraints to python because of ConstraintFunction.
struct BasicConstraints4py
{
    /// The coefficient matrix of the linear equality constraint equations \eq{Ax=b}.
    Matrix A;

    /// The constraint function in the non-linear equality constraint equations \eq{h(x) = 0}.
    ConstraintFunction4py h;

    /// The indices of the variables with lower bounds.
    Indices ilower;

    /// The indices of the variables with upper bounds.
    Indices iupper;

    /// The indices of the variables with fixed values.
    Indices ifixed;
};

/// A workaround to expose BasicProblem to python because of ObjectiveFunction.
struct BasicProblem4py
{
    /// The dimensions of the basic optimization problem.
    BasicDims dims;

    /// The constraints of the basic optimization problem.
    BasicConstraints4py constraints;

    /// The objective function of the basic optimization problem.
    ObjectiveFunction4py objective;
};

/// Convert an ObjectiveFunction4py function to an ObjectiveFunction function.
auto convert(const ObjectiveFunction4py& obj4py) -> ObjectiveFunction
{
    if(obj4py == nullptr)
        return {};

    return [=](VectorConstRef x, ObjectiveResult& res)
    {
        ObjectiveResult4py res4py(res);
        obj4py(x, &res4py);
        res.f = res4py.f;
        res.failed = res4py.failed;
    };
}

/// Convert a ConstraintFunction4py function to a ConstraintFunction function.
auto convert(const ConstraintFunction4py& constraintfn4py) -> ConstraintFunction
{
    if(constraintfn4py == nullptr)
        return {};

    return [=](VectorConstRef x, ConstraintResult& res)
    {
        ConstraintResult4py res4py(res);
        constraintfn4py(x, &res4py);
        res.failed = res4py.failed;
    };
}

/// Convert a BasicConstraints4py object to a BasicConstraints object.
auto convert(const BasicConstraints4py& constraints4py)
{
    return BasicConstraints{
        constraints4py.A,
        convert(constraints4py.h),
        constraints4py.ilower,
        constraints4py.iupper,
        constraints4py.ifixed
    };
}

/// Convert a BasicProblem4py object to a BasicProblem object.
auto convert(const BasicProblem4py& problem4py)
{
    return BasicProblem{
        problem4py.dims,
        convert(problem4py.constraints),
        convert(problem4py.objective)
    };
}

void exportBasicSolver(py::module& m)
{
    py::class_<BasicState>(m, "BasicState")
        .def(py::init<>())
        .def(py::init<Index, Index>(), "Construct a BasicState object with given dimension values.", py::arg("n"), py::arg("m"))
        .def_readwrite("x", &BasicState::x, "The primal variables of the basic optimization problem.")
        .def_readwrite("y", &BasicState::y, "The Lagrange multipliers with respect to the equality constraints Ax = b and h(x) = 0.")
        .def_readwrite("z", &BasicState::z, "The slack variables with respect to the lower bounds of the primal variables.")
        .def_readwrite("w", &BasicState::w, "The slack variables with respect to the upper bounds of the primal variables.")
        ;

    py::class_<BasicDims>(m, "BasicDims")
        .def(py::init<>())
        .def_readwrite("x", &BasicDims::x, "The number of variables (equivalent to the dimension of vector x).")
        .def_readwrite("b", &BasicDims::b, "The number of linear equality constraint equations (equivalent to the dimension of vector b).")
        .def_readwrite("h", &BasicDims::h, "The number of non-linear equality constraint equations (equivalent to the dimension of vector h).")
        .def_readwrite("xlower", &BasicDims::xlower, "The number of variables with lower bounds (equivalent to the dimension of vector of lower bounds xl).")
        .def_readwrite("xupper", &BasicDims::xupper, "The number of variables with upper bounds (equivalent to the dimension of vector of upper bounds xu).")
        .def_readwrite("xfixed", &BasicDims::xfixed, "The number of variables with fixed values (equivalent to the dimension of vector xf).")
        ;

    py::class_<BasicConstraints4py>(m, "BasicConstraints")
        .def(py::init<>())
        .def_readwrite("A", &BasicConstraints4py::A, "The coefficient matrix of the linear equality constraint equations Ax = b.")
        .def_readwrite("h", &BasicConstraints4py::h, "The constraint function in the non-linear equality constraint equations h(x) = 0.")
        .def_readwrite("ilower", &BasicConstraints4py::ilower, "The indices of the variables with lower bounds.")
        .def_readwrite("iupper", &BasicConstraints4py::iupper, "The indices of the variables with upper bounds.")
        .def_readwrite("ifixed", &BasicConstraints4py::ifixed, "The indices of the variables with fixed values.")
        ;

    py::class_<BasicParams>(m, "BasicParams")
        .def(py::init<>())
        .def_readwrite("b", &BasicParams::b, "The right-hand side vector of the linear equality constraints Ax = b.")
        .def_readwrite("xlower", &BasicParams::xlower, "The lower bounds of the variables in x that have lower bounds.")
        .def_readwrite("xupper", &BasicParams::xupper, "The upper bounds of the variables x that have upper bounds.")
        .def_readwrite("xfixed", &BasicParams::xfixed, "The values of the variables in x that are fixed.")
        .def_readwrite("extra", &BasicParams::extra, "The extra parameters in the problem.")
        ;

    py::class_<BasicProblem4py>(m, "BasicProblem")
        .def(py::init<>())
        .def_readwrite("dims", &BasicProblem4py::dims, "The dimensions of the basic optimization problem.")
        .def_readwrite("constraints", &BasicProblem4py::constraints, "The constraints of the basic optimization problem.")
        .def_readwrite("objective", &BasicProblem4py::objective, "The objective function of the basic optimization problem.")
        ;

	// A constructor function for BasicSolver that accepts BasicProblem4py instead of BasicProblem.
    auto initBasicSolver = [](const BasicProblem4py& problem4py)
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
