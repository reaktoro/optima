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
#include <Optima/Echelonizer.hpp>
using namespace Optima;

void exportEchelonizer(py::module& m)
{
    auto init = [](MatrixView4py A) -> Echelonizer
    {
        return Echelonizer(A);
    };

    auto compute = [](Echelonizer& self, MatrixView4py A)
    {
        return self.compute(A);
    };

    py::class_<Echelonizer>(m, "Echelonizer")
        .def(py::init<>())
        .def(py::init<const Echelonizer&>())
        .def(py::init(init))
        .def("numVariables", &Echelonizer::numVariables)
        .def("numEquations", &Echelonizer::numEquations)
        .def("numBasicVariables", &Echelonizer::numBasicVariables)
        .def("numNonBasicVariables", &Echelonizer::numNonBasicVariables)
        .def("S", &Echelonizer::S, py::return_value_policy::reference_internal)
        .def("R", &Echelonizer::R, py::return_value_policy::reference_internal)
        .def("Q", &Echelonizer::Q, py::return_value_policy::reference_internal)
        .def("C", &Echelonizer::C)
        .def("indicesEquations", &Echelonizer::indicesEquations)
        .def("indicesBasicVariables", &Echelonizer::indicesBasicVariables, py::return_value_policy::reference_internal)
        .def("indicesNonBasicVariables", &Echelonizer::indicesNonBasicVariables, py::return_value_policy::reference_internal)
        .def("compute", compute)
        .def("updateWithSwapBasicVariable", &Echelonizer::updateWithSwapBasicVariable)
        .def("updateWithPriorityWeights", &Echelonizer::updateWithPriorityWeights)
        .def("updateOrdering", &Echelonizer::updateOrdering)
        .def("reset", &Echelonizer::reset)
        .def("cleanResidualRoundoffErrors", &Echelonizer::cleanResidualRoundoffErrors)
        ;
}
