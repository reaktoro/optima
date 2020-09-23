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
        .def(py::init<>())
        .def_readwrite("f", &ObjectiveRequirement::f, "The flag indicating if the objective function f(x, p) needs to be evaluated.")
        .def_readwrite("fx", &ObjectiveRequirement::fx, "The flag indicating if the gradient function fx(x, p) needs to be evaluated.")
        .def_readwrite("fxx", &ObjectiveRequirement::fxx, "The flag indicating if the Jacobian function fxx(x, p) needs to be evaluated.")
        .def_readwrite("fxp", &ObjectiveRequirement::fxp, "The flag indicating if the Jacobian function fxp(x, p) needs to be evaluated.")
        ;

    py::class_<ObjectiveResult4py>(m, "ObjectiveResult")
        .def_readwrite("f", &ObjectiveResult4py::f, "The evaluated objective function f(x, p).")
        .def_readwrite("fx", &ObjectiveResult4py::fx, "The evaluated gradient of the objective function f(x, p) with respect to x.")
        .def_readwrite("fxx", &ObjectiveResult4py::fxx, "The evaluated Jacobian of the gradient function fx(x, p) with respect to x, i.e., the Hessian of f(x, p) with respect to x.")
        .def_readwrite("fxp", &ObjectiveResult4py::fxp, "The evaluated Jacobian of the gradient function fx(x, p) with respect to p.")
        .def_readwrite("requires", &ObjectiveResult4py::requires, "The requirements in the evaluation of the objective function.")
        .def_readwrite("failed", &ObjectiveResult4py::failed, "The boolean flag that indicates if the objective function evaluation failed.")
        ;
}
