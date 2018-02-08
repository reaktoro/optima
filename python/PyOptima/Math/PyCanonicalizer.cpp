// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2017 Allan Leal
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
#include <Optima/Math/Canonicalizer.hpp>
using namespace Optima;

void exportCanonicalizer(py::module& m)
{
    py::class_<Canonicalizer>(m, "Canonicalizer")
        .def(py::init<>())
        .def(py::init<MatrixXdConstRef>())
        .def("numVariables", &Canonicalizer::numVariables)
        .def("numEquations", &Canonicalizer::numEquations)
        .def("numBasicVariables", &Canonicalizer::numBasicVariables)
        .def("numNonBasicVariables", &Canonicalizer::numNonBasicVariables)
        .def("S", &Canonicalizer::S, py::return_value_policy::reference_internal)
        .def("R", &Canonicalizer::R, py::return_value_policy::reference_internal)
        .def("Q", &Canonicalizer::Q, py::return_value_policy::reference_internal)
        .def("C", &Canonicalizer::C)
        .def("ili", &Canonicalizer::ili)
        .def("ibasic", &Canonicalizer::ibasic, py::return_value_policy::reference_internal)
        .def("inonbasic", &Canonicalizer::inonbasic, py::return_value_policy::reference_internal)
        .def("compute", &Canonicalizer::compute)
        .def("rationalize", &Canonicalizer::rationalize)
        .def("swapBasicVariable", &Canonicalizer::swapBasicVariable)
        .def("update", &Canonicalizer::update)
        .def("reorder", &Canonicalizer::reorder)
        ;
}
