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
#include <Optima/CanonicalizerAdvanced.hpp>
using namespace Optima;

void exportCanonicalizerAdvanced(py::module& m)
{
    py::class_<CanonicalizerAdvanced>(m, "CanonicalizerAdvanced")
        .def(py::init<>())
        .def(py::init<MatrixConstRef, MatrixConstRef>())
        .def("numVariables", &CanonicalizerAdvanced::numVariables)
        .def("numEquations", &CanonicalizerAdvanced::numEquations)
        .def("numBasicVariables", &CanonicalizerAdvanced::numBasicVariables)
        .def("numNonBasicVariables", &CanonicalizerAdvanced::numNonBasicVariables)
        .def("S", &CanonicalizerAdvanced::S, py::return_value_policy::reference_internal)
        .def("R", &CanonicalizerAdvanced::R, py::return_value_policy::reference_internal)
        .def("Q", &CanonicalizerAdvanced::Q, py::return_value_policy::reference_internal)
        .def("C", &CanonicalizerAdvanced::C)
        .def("indicesBasicVariables", &CanonicalizerAdvanced::indicesBasicVariables, py::return_value_policy::reference_internal)
        .def("indicesNonBasicVariables", &CanonicalizerAdvanced::indicesNonBasicVariables, py::return_value_policy::reference_internal)
        .def("compute", &CanonicalizerAdvanced::compute)
        .def("updateWithPriorityWeights", &CanonicalizerAdvanced::updateWithPriorityWeights)
        ;
}
