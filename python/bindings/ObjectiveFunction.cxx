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
    // TODO: Instead of having an extra type ObjectiveResult4py because of double& and bool&,
    // consider using only ObjectiveResult and def_property for f and diagxx members.
    py::class_<ObjectiveResult4py>(m, "ObjectiveResult")
        .def_readwrite("f", &ObjectiveResult4py::f, "The evaluated objective function f(x, p).")
        .def_readwrite("fx", &ObjectiveResult4py::fx, "The evaluated gradient of f(x, p) with respect to x.")
        .def_readwrite("fxx", &ObjectiveResult4py::fxx, "The evaluated Jacobian fx(x, p) with respect to x.")
        .def_readwrite("fxp", &ObjectiveResult4py::fxp, "The evaluated Jacobian fx(x, p) with respect to p.")
        .def_readwrite("diagfxx", &ObjectiveResult4py::diagfxx, "The flag indicating whether `fxx` is diagonal.")
        ;
}
