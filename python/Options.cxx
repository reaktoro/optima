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
#include <pybind11/stl.h>
namespace py = pybind11;

// Optima includes
#include <Optima/Options.hpp>
using namespace Optima;

void exportOptions(py::module& m)
{
    py::enum_<StepMode>(m, "StepMode")
        .value("Conservative", StepMode::Conservative)
        .value("Aggressive", StepMode::Aggressive)
        ;

    py::class_<OutputOptions, OutputterOptions>(m, "OutputOptions")
        .def(py::init<>())
        .def_readwrite("xprefix", &OutputOptions::xprefix)
        .def_readwrite("yprefix", &OutputOptions::yprefix)
        .def_readwrite("zprefix", &OutputOptions::zprefix)
        .def_readwrite("xnames", &OutputOptions::xnames)
        .def_readwrite("ynames", &OutputOptions::ynames)
        ;

    py::class_<Options>(m, "Options")
        .def(py::init<>())
        .def_readwrite("output", &Options::output)
        .def_readwrite("tolerance", &Options::tolerance)
        .def_readwrite("tolerancex", &Options::tolerancex)
        .def_readwrite("tolerancef", &Options::tolerancef)
        .def_readwrite("max_iterations", &Options::max_iterations)
        .def_readwrite("mu", &Options::mu)
        .def_readwrite("tau", &Options::tau)
        .def_readwrite("step", &Options::step)
        .def_readwrite("kkt", &Options::kkt)
        ;
}
