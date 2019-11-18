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
#include <Optima/Structure.hpp>
using namespace Optima;

void exportStructure(py::module& m)
{
    py::class_<Structure>(m, "Structure")
        .def(py::init<Index>(), py::arg("n"), "Construct a Structure object with given number of variables")
        .def("setEqualityConstraintMatrix", &Structure::setEqualityConstraintMatrix, "Set the equality constraint matrix Ae.")
        .def("setInequalityConstraintMatrix", &Structure::setInequalityConstraintMatrix, "Set the inequality constraint matrix Ai.")
        .def("setVariablesWithLowerBounds", &Structure::setVariablesWithLowerBounds, "Set the indices of the variables in `x` with lower bounds.")
        .def("allVariablesHaveLowerBounds", &Structure::allVariablesHaveLowerBounds, "Set all variables in `x` with lower bounds.")
        .def("setVariablesWithUpperBounds", &Structure::setVariablesWithUpperBounds, "Set the indices of the variables in `x` with upper bounds.")
        .def("allVariablesHaveUpperBounds", &Structure::allVariablesHaveUpperBounds, "Set all variables in `x` with upper bounds.")
        .def("setVariablesWithFixedValues", &Structure::setVariablesWithFixedValues, "Set the indices of the variables in `x` with fixed values.")
        .def("numVariables", &Structure::numVariables, "Return the number of variables.")
        .def("numEqualityConstraints", &Structure::numEqualityConstraints, "Return the number of linear equality constraints.")
        .def("numInequalityConstraints", &Structure::numInequalityConstraints, "Return the number of linear inequality constraints.")
        .def("equalityConstraintMatrix", &Structure::equalityConstraintMatrix, py::return_value_policy::reference_internal, "Return the equality constraint matrix Ae.")
        .def("inequalityConstraintMatrix", &Structure::inequalityConstraintMatrix, py::return_value_policy::reference_internal, "Return the inequality constraint matrix Ai.")
        .def("variablesWithLowerBounds", &Structure::variablesWithLowerBounds, py::return_value_policy::reference_internal, "Return the indices of the variables with lower bounds.")
        .def("variablesWithUpperBounds", &Structure::variablesWithUpperBounds, py::return_value_policy::reference_internal, "Return the indices of the variables with upper bounds.")
        .def("variablesWithFixedValues", &Structure::variablesWithFixedValues, py::return_value_policy::reference_internal, "Return the indices of the variables with fixed values.")
        .def("variablesWithoutLowerBounds", &Structure::variablesWithoutLowerBounds, py::return_value_policy::reference_internal, "Return the indices of the variables without lower bounds.")
        .def("variablesWithoutUpperBounds", &Structure::variablesWithoutUpperBounds, py::return_value_policy::reference_internal, "Return the indices of the variables without upper bounds.")
        .def("variablesWithoutFixedValues", &Structure::variablesWithoutFixedValues, py::return_value_policy::reference_internal, "Return the indices of the variables without fixed values.")
        .def("orderingLowerBounds", &Structure::orderingLowerBounds, py::return_value_policy::reference_internal, "Return the indices of the variables without fixed values.")
        .def("orderingUpperBounds", &Structure::orderingUpperBounds, py::return_value_policy::reference_internal, "Return the indices of the variables without fixed values.")
        .def("orderingFixedValues", &Structure::orderingFixedValues, py::return_value_policy::reference_internal, "Return the indices of the variables without fixed values.")
        ;
}
