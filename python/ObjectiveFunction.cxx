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
#include <Optima/ObjectiveFunction.hpp>
using namespace Optima;

void exportObjectiveFunction(py::module& m)
{
    py::class_<ObjectiveRequirement>(m, "ObjectiveRequirement")
        .def_readwrite("f", &ObjectiveRequirement::f, "The boolean flag that indicates the need for the objective value.")
        .def_readwrite("g", &ObjectiveRequirement::g, "The boolean flag that indicates the need for the objective gradient.")
        .def_readwrite("H", &ObjectiveRequirement::H, "The boolean flag that indicates the need for the objective Hessian.")
        ;

    py::class_<ObjectiveResult>(m, "ObjectiveResult")
        .def(py::init<VectorRef, MatrixRef>())
        .def_readwrite("f", &ObjectiveResult::f, "The evaluated value of the objective function.")
        .def_readwrite("g", &ObjectiveResult::g, "The evaluated gradient of the objective function.")
        .def_readwrite("H", &ObjectiveResult::H, "The evaluated Hessian of the objective function.")
        .def_readwrite("requires", &ObjectiveResult::requires, "The requirements in the evaluation of the objective function.")
        .def_readwrite("failed", &ObjectiveResult::failed, "The boolean flag that indicates if the objective function evaluation failed.")
        ;
}
