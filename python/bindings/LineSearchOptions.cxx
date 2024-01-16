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
#include <Optima/LineSearchOptions.hpp>
using namespace Optima;

void exportLineSearchOptions(py::module& m)
{
    py::class_<LineSearchOptions>(m, "LineSearchOptions")
        .def(py::init<>())
        .def_readwrite("tolerance", &LineSearchOptions::tolerance)
        .def_readwrite("maxiterations", &LineSearchOptions::maxiterations)
        .def_readwrite("trigger_when_current_error_is_greater_than_initial_error_by_factor", &LineSearchOptions::trigger_when_current_error_is_greater_than_initial_error_by_factor)
        .def_readwrite("trigger_when_current_error_is_greater_than_previous_error_by_factor", &LineSearchOptions::trigger_when_current_error_is_greater_than_previous_error_by_factor)
        ;
}
