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
#include <Optima/Constraints.hpp>
using namespace Optima;

void exportConstraints(py::module& m)
{
    py::class_<Constraints>(m, "Constraints")
        .def(py::init<Index>(), py::arg("n"), "Construct a Constraints object with given number of variables")
        .def("setEqualityConstraintMatrix", &Constraints::setEqualityConstraintMatrix, "Set the linear equality constraint matrix Ae.")
        .def("setEqualityConstraintFunction", &Constraints::setEqualityConstraintFunction, "Set the non-linear equality constraint function he(x).")
        .def("setInequalityConstraintMatrix", &Constraints::setInequalityConstraintMatrix, "Set the linear inequality constraint matrix Ai.")
        .def("setInequalityConstraintFunction", &Constraints::setInequalityConstraintFunction, "Set the non-linear inequality constraint function hi(x).")
        .def("setVariablesWithLowerBounds", &Constraints::setVariablesWithLowerBounds, "Set the indices of the variables in `x` with lower bounds.")
        .def("allVariablesHaveLowerBounds", &Constraints::allVariablesHaveLowerBounds, "Set all variables in `x` with lower bounds.")
        .def("setVariablesWithUpperBounds", &Constraints::setVariablesWithUpperBounds, "Set the indices of the variables in `x` with upper bounds.")
        .def("allVariablesHaveUpperBounds", &Constraints::allVariablesHaveUpperBounds, "Set all variables in `x` with upper bounds.")
        .def("setVariablesWithFixedValues", &Constraints::setVariablesWithFixedValues, "Set the indices of the variables in `x` with fixed values.")
        .def("numVariables", &Constraints::numVariables, "Return the number of variables.")
        .def("numLinearEqualityConstraints", &Constraints::numLinearEqualityConstraints, "Return the number of linear equality constraints.")
        .def("numLinearInequalityConstraints", &Constraints::numLinearInequalityConstraints, "Return the number of linear inequality constraints.")
        .def("numNonLinearEqualityConstraints", &Constraints::numNonLinearEqualityConstraints, "Return the number of non-linear equality constraints.")
        .def("numNonLinearInequalityConstraints", &Constraints::numNonLinearInequalityConstraints, "Return the number of non-linear inequality constraints.")
        .def("equalityConstraintMatrix", &Constraints::equalityConstraintMatrix, py::return_value_policy::reference_internal, "Return the equality constraint matrix Ae.")
        .def("equalityConstraintFunction", &Constraints::equalityConstraintFunction, py::return_value_policy::reference_internal, "Return the equality constraint function he.")
        .def("inequalityConstraintMatrix", &Constraints::inequalityConstraintMatrix, py::return_value_policy::reference_internal, "Return the inequality constraint matrix Ai.")
        .def("inequalityConstraintFunction", &Constraints::inequalityConstraintFunction, py::return_value_policy::reference_internal, "Return the inequality constraint function hi.")
        .def("variablesWithLowerBounds", &Constraints::variablesWithLowerBounds, py::return_value_policy::reference_internal, "Return the indices of the variables with lower bounds.")
        .def("variablesWithUpperBounds", &Constraints::variablesWithUpperBounds, py::return_value_policy::reference_internal, "Return the indices of the variables with upper bounds.")
        .def("variablesWithFixedValues", &Constraints::variablesWithFixedValues, py::return_value_policy::reference_internal, "Return the indices of the variables with fixed values.")
        .def("variablesWithoutLowerBounds", &Constraints::variablesWithoutLowerBounds, py::return_value_policy::reference_internal, "Return the indices of the variables without lower bounds.")
        .def("variablesWithoutUpperBounds", &Constraints::variablesWithoutUpperBounds, py::return_value_policy::reference_internal, "Return the indices of the variables without upper bounds.")
        .def("variablesWithoutFixedValues", &Constraints::variablesWithoutFixedValues, py::return_value_policy::reference_internal, "Return the indices of the variables without fixed values.")
        .def("orderingLowerBounds", &Constraints::orderingLowerBounds, py::return_value_policy::reference_internal, "Return the indices of the variables without fixed values.")
        .def("orderingUpperBounds", &Constraints::orderingUpperBounds, py::return_value_policy::reference_internal, "Return the indices of the variables without fixed values.")
        .def("orderingFixedValues", &Constraints::orderingFixedValues, py::return_value_policy::reference_internal, "Return the indices of the variables without fixed values.")
        ;
}
