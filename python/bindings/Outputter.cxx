// Optima is a C++ library for solving linear and non-linear constrained optimization problems
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
#include "pybind11.hxx"

// Optima includes
#include <Optima/Outputter.hpp>
using namespace Optima;

void exportOutputter(py::module& m)
{
    py::class_<OutputterOptions>(m, "OutputterOptions")
        .def_readwrite("active", &OutputterOptions::active)
        .def_readwrite("fixed", &OutputterOptions::fixed)
        .def_readwrite("scientific", &OutputterOptions::scientific)
        .def_readwrite("precision", &OutputterOptions::precision)
        .def_readwrite("width", &OutputterOptions::width)
        .def_readwrite("separator", &OutputterOptions::separator)
        ;
}
