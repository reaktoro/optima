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
#include <Optima/OptimumStructure.hpp>
using namespace Optima;

void exportOptimumStructure(py::module& m)
{
    py::class_<OptimumStructure>(m, "OptimumStructure")
        .def_readwrite("A", &OptimumStructure::A, "The coefficient matrix A of the linear equality constraints.")
        .def(py::init<Index, Index>(), py::arg("n"), py::arg("m"))
        .def("setVariablesWithLowerBounds", &OptimumStructure::setVariablesWithLowerBounds, "Set the indices of the variables in `x` with lower bounds.")
        .def("allVariablesHaveLowerBounds", &OptimumStructure::allVariablesHaveLowerBounds, "Set all variables in `x` with lower bounds.")
        .def("setVariablesWithUpperBounds", &OptimumStructure::setVariablesWithUpperBounds, "Set the indices of the variables in `x` with upper bounds.")
        .def("allVariablesHaveUpperBounds", &OptimumStructure::allVariablesHaveUpperBounds, "Set all variables in `x` with upper bounds.")
        .def("setVariablesWithFixedValues", &OptimumStructure::setVariablesWithFixedValues, "Set the indices of the variables in `x` with fixed values.")
        .def("numVariables", &OptimumStructure::numVariables, py::return_value_policy::reference_internal, "Return the number of variables.")
        .def("numEqualityConstraints", &OptimumStructure::numEqualityConstraints, py::return_value_policy::reference_internal, "Return the number of linear equality constraints.")
        .def("variablesWithLowerBounds", &OptimumStructure::variablesWithLowerBounds, py::return_value_policy::reference_internal, "Return the indices of the variables with lower bounds.")
        .def("variablesWithUpperBounds", &OptimumStructure::variablesWithUpperBounds, py::return_value_policy::reference_internal, "Return the indices of the variables with upper bounds.")
        .def("variablesWithFixedValues", &OptimumStructure::variablesWithFixedValues, py::return_value_policy::reference_internal, "Return the indices of the variables with fixed values.")
        .def("variablesWithoutLowerBounds", &OptimumStructure::variablesWithoutLowerBounds, py::return_value_policy::reference_internal, "Return the indices of the variables without lower bounds.")
        .def("variablesWithoutUpperBounds", &OptimumStructure::variablesWithoutUpperBounds, py::return_value_policy::reference_internal, "Return the indices of the variables without upper bounds.")
        .def("variablesWithoutFixedValues", &OptimumStructure::variablesWithoutFixedValues, py::return_value_policy::reference_internal, "Return the indices of the variables without fixed values.")
        .def("orderingLowerBounds", &OptimumStructure::orderingLowerBounds, py::return_value_policy::reference_internal, "Return the indices of the variables without fixed values.")
        .def("orderingUpperBounds", &OptimumStructure::orderingUpperBounds, py::return_value_policy::reference_internal, "Return the indices of the variables without fixed values.")
        .def("orderingFixedValues", &OptimumStructure::orderingFixedValues, py::return_value_policy::reference_internal, "Return the indices of the variables without fixed values.")
        ;
}
