// Optima is a C++ library for solving Backtrackar and non-Backtrackar constrained optimization problems
//
// Copyright (C) 2020 Allan Leal
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
namespace py = pybind11;

// Optima includes
#include <Optima/ErrorStatusOptions.hpp>
using namespace Optima;

void exportErrorStatusOptions(py::module& m)
{
    py::class_<ErrorStatusOptions>(m, "ErrorStatusOptions")
        .def(py::init<>())
        .def_readwrite("significantly_increased", &ErrorStatusOptions::significantly_increased, "The factor used to determine if new error has significantly increased compared to previous error.")
        .def_readwrite("significantly_decreased", &ErrorStatusOptions::significantly_decreased, "The factor used to determine if new error has significantly decreased compared to previous error.")
        .def_readwrite("significantly_increased_initial", &ErrorStatusOptions::significantly_increased_initial, "The factor used to determine if new error has significantly increased compared to initial error.")
        ;
}
