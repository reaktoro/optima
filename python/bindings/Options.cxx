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

    py::class_<LineSearchOptions>(m, "LineSearchOptions")
        .def_readwrite("tolerance", &LineSearchOptions::tolerance)
        .def_readwrite("maxiters", &LineSearchOptions::maxiters)
        .def_readwrite("trigger_when_current_error_is_greater_than_initial_error_by_factor", &LineSearchOptions::trigger_when_current_error_is_greater_than_initial_error_by_factor)
        .def_readwrite("trigger_when_current_error_is_greater_than_previous_error_by_factor", &LineSearchOptions::trigger_when_current_error_is_greater_than_previous_error_by_factor)
        ;

    py::class_<BacktrackSearchOptions>(m, "BacktrackSearchOptions")
        .def_readwrite("factor", &BacktrackSearchOptions::factor)
        .def_readwrite("maxiters", &BacktrackSearchOptions::maxiters)
        ;

    py::class_<SteepestDescentOptions>(m, "SteepestDescentOptions")
        .def_readwrite("tolerance", &SteepestDescentOptions::tolerance)
        .def_readwrite("maxiters", &SteepestDescentOptions::maxiters)
        ;

    py::class_<Options>(m, "Options")
        .def(py::init<>())
        .def_readwrite("output", &Options::output)
        .def_readwrite("maxiterations", &Options::maxiterations)
        .def_readwrite("kkt", &Options::kkt)
        .def_readwrite("linesearch", &Options::linesearch)
        .def_readwrite("steepestdescent", &Options::steepestdescent)
        .def_readwrite("backtrack", &Options::backtrack)
        ;
}
