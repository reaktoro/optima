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
#include <Optima/StablePartition.hpp>
using namespace Optima;

void exportStablePartition(py::module& m)
{
    py::class_<StablePartition>(m, "StablePartition")
        .def(py::init<Index>())
        .def("setStable", &StablePartition::setStable)
        .def("setUnstable", &StablePartition::setUnstable)
        .def("stable", &StablePartition::stable, py::return_value_policy::reference_internal)
        .def("unstable", &StablePartition::unstable, py::return_value_policy::reference_internal)
        ;
}
