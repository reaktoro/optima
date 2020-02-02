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
#include <Optima/ExtendedCanonicalizer.hpp>
using namespace Optima;

void exportExtendedCanonicalizer(py::module& m)
{
    auto init = [](MatrixConstRef4py A) -> ExtendedCanonicalizer
    {
        return ExtendedCanonicalizer(A);
    };

    auto updateWithPriorityWeights = [](ExtendedCanonicalizer& self, MatrixConstRef4py J, VectorConstRef weights)
    {
        return self.updateWithPriorityWeights(J, weights);
    };

    py::class_<ExtendedCanonicalizer>(m, "ExtendedCanonicalizer")
        .def(py::init<>())
        .def(py::init(init))
        .def("numVariables", &ExtendedCanonicalizer::numVariables)
        .def("numEquations", &ExtendedCanonicalizer::numEquations)
        .def("numBasicVariables", &ExtendedCanonicalizer::numBasicVariables)
        .def("numNonBasicVariables", &ExtendedCanonicalizer::numNonBasicVariables)
        .def("S", &ExtendedCanonicalizer::S, py::return_value_policy::reference_internal)
        .def("R", &ExtendedCanonicalizer::R, py::return_value_policy::reference_internal)
        .def("Q", &ExtendedCanonicalizer::Q, py::return_value_policy::reference_internal)
        .def("C", &ExtendedCanonicalizer::C)
        .def("indicesBasicVariables", &ExtendedCanonicalizer::indicesBasicVariables, py::return_value_policy::reference_internal)
        .def("indicesNonBasicVariables", &ExtendedCanonicalizer::indicesNonBasicVariables, py::return_value_policy::reference_internal)
        .def("updateWithPriorityWeights", updateWithPriorityWeights)
        ;
}
