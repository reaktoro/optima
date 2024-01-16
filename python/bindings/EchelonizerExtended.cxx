// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
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
#include "pybind11.hxx"

// Optima includes
#include <Optima/EchelonizerExtended.hpp>
using namespace Optima;

void exportEchelonizerExtended(py::module& m)
{
    auto init = [](MatrixView4py A) -> EchelonizerExtended
    {
        return EchelonizerExtended(A);
    };

    auto updateWithPriorityWeights = [](EchelonizerExtended& self, MatrixView4py J, VectorView weights)
    {
        return self.updateWithPriorityWeights(J, weights);
    };

    py::class_<EchelonizerExtended>(m, "EchelonizerExtended")
        .def(py::init<>())
        .def(py::init(init))
        .def("numVariables", &EchelonizerExtended::numVariables)
        .def("numEquations", &EchelonizerExtended::numEquations)
        .def("numBasicVariables", &EchelonizerExtended::numBasicVariables)
        .def("numNonBasicVariables", &EchelonizerExtended::numNonBasicVariables)
        .def("S", &EchelonizerExtended::S, py::return_value_policy::reference_internal)
        .def("R", &EchelonizerExtended::R, py::return_value_policy::reference_internal)
        .def("Q", &EchelonizerExtended::Q, py::return_value_policy::reference_internal)
        .def("C", &EchelonizerExtended::C)
        .def("indicesBasicVariables", &EchelonizerExtended::indicesBasicVariables, py::return_value_policy::reference_internal)
        .def("indicesNonBasicVariables", &EchelonizerExtended::indicesNonBasicVariables, py::return_value_policy::reference_internal)
        .def("updateWithPriorityWeights", updateWithPriorityWeights)
        .def("updateOrdering", &EchelonizerExtended::updateOrdering)
        .def("cleanResidualRoundoffErrors", &EchelonizerExtended::cleanResidualRoundoffErrors)
        ;
}
