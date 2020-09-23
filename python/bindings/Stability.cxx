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
#include <Optima/Stability.hpp>
using namespace Optima;

void exportStability(py::module& m)
{
    py::class_<Stability::Data>(m, "StabilityData")
        .def(py::init<>())
        .def_readwrite("iordering", &Stability::Data::iordering)
        .def_readwrite("ns", &Stability::Data::ns)
        .def_readwrite("nlu", &Stability::Data::nlu)
        .def_readwrite("nuu", &Stability::Data::nuu)
        .def_readwrite("nslu", &Stability::Data::nslu)
        .def_readwrite("nsuu", &Stability::Data::nsuu)
        ;

    py::class_<Stability>(m, "Stability")
        .def(py::init<>())
        .def(py::init<const Stability::Data&>())
        .def("update", &Stability::update)
        .def("numVariables", &Stability::numVariables)
        .def("numStableVariables", &Stability::numStableVariables)
        .def("numUnstableVariables", &Stability::numUnstableVariables)
        .def("numLowerUnstableVariables", &Stability::numLowerUnstableVariables)
        .def("numUpperUnstableVariables", &Stability::numUpperUnstableVariables)
        .def("numStrictlyLowerUnstableVariables", &Stability::numStrictlyLowerUnstableVariables)
        .def("numStrictlyUpperUnstableVariables", &Stability::numStrictlyUpperUnstableVariables)
        .def("numStrictlyUnstableVariables", &Stability::numStrictlyUnstableVariables)
        .def("indicesVariables", &Stability::indicesVariables, py::return_value_policy::reference_internal)
        .def("indicesStableVariables", &Stability::indicesStableVariables, py::return_value_policy::reference_internal)
        .def("indicesUnstableVariables", &Stability::indicesUnstableVariables, py::return_value_policy::reference_internal)
        .def("indicesLowerUnstableVariables", &Stability::indicesLowerUnstableVariables, py::return_value_policy::reference_internal)
        .def("indicesUpperUnstableVariables", &Stability::indicesUpperUnstableVariables, py::return_value_policy::reference_internal)
        .def("indicesStrictlyLowerUnstableVariables", &Stability::indicesStrictlyLowerUnstableVariables, py::return_value_policy::reference_internal)
        .def("indicesStrictlyUpperUnstableVariables", &Stability::indicesStrictlyUpperUnstableVariables, py::return_value_policy::reference_internal)
        .def("indicesStrictlyUnstableVariables", &Stability::indicesStrictlyUnstableVariables, py::return_value_policy::reference_internal)
        ;
}
