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
#include <Optima/Problem.hpp>
using namespace Optima;

void exportProblem(py::module& m)
{
    py::class_<Dims>(m, "Dims")
        .def(py::init<>())
        .def_readwrite("x", &Dims::x)
        .def_readwrite("be", &Dims::be)
        .def_readwrite("bg", &Dims::bg)
        .def_readwrite("he", &Dims::he)
        .def_readwrite("hg", &Dims::hg)
        ;

    // using ObjectivePtr = std::function<void(VectorConstRef, ObjectiveResult*)>;

	// // This is a workaround to let Python callback change the state of ObjectiveResult, and not a copy
    // auto createProblem = [](const ObjectivePtr& pyobjective, const Constraints& constraints)
    // {
    //     ObjectiveFunction objective = [=](VectorConstRef x, ObjectiveResult& f) { pyobjective(x, &f); };
    //     return Problem(objective, constraints);
    // };

	// // This is a workaround to let Python callback change the state of ObjectiveResult, and not a copy
	// auto get_objective = [](const Problem& self)
	// {
    //     auto obj = self.objective();
	// 	return [=](VectorConstRef x, ObjectiveResult* f) { obj(x, *f); };
	// };

    // py::class_<Problem>(m, "Problem")
    //     .def(py::init(createProblem), "Construct a Problem instance with given objective and constraints.")
    //     .def("setEqualityConstraintVector", &Problem::setEqualityConstraintVector, "Set the right-hand side vector be of the equality constraint equation Ae x = be.")
    //     .def("setInequalityConstraintVector", &Problem::setInequalityConstraintVector, "Set the right-hand side vector bi of the equality constraint equation Ai x >= bi.")
    //     .def("setLowerBound", &Problem::setLowerBound, "Set a common lower bound value for all variables in x that have lower bounds.")
    //     .def("setLowerBounds", &Problem::setLowerBounds, "Set the lower bound values for all variables in x that have lower bounds.")
    //     .def("setUpperBound", &Problem::setUpperBound, "Set a common upper bound value for all variables in x that have upper bounds.")
    //     .def("setUpperBounds", &Problem::setUpperBounds, "Set the upper bound values for all variables in x that have upper bounds.")
    //     .def("setFixedValue", &Problem::setFixedValue, "Set a common fixed value for all variables in x that have fixed values.")
    //     .def("setFixedValues", &Problem::setFixedValues, "Set the fixed values of all variables in x that have fixed values.")
    //     .def("equalityConstraintVector", &Problem::equalityConstraintVector, py::return_value_policy::reference_internal, "Return right-hand side vector be of the equality constraint equation Ae x = be.")
    //     .def("inequalityConstraintVector", &Problem::inequalityConstraintVector, py::return_value_policy::reference_internal, "Return the right-hand side vector bi of the equality constraint equation Aix >= bi.")
    //     .def("lowerBounds", &Problem::lowerBounds, py::return_value_policy::reference_internal, "Return the lower bound values of the variables in x that have lower bounds.")
    //     .def("upperBounds", &Problem::upperBounds, py::return_value_policy::reference_internal, "Return the upper bound values of the variables in x that have upper bounds.")
    //     .def("fixedValues", &Problem::fixedValues, py::return_value_policy::reference_internal, "Return the fixed values of the variables in x that have fixed values.")
    //     ;
}
