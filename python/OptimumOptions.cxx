// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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
#include <Optima/OptimumOptions.hpp>
using namespace Optima;

void exportOptimumOptions(py::module& m)
{
    py::enum_<StepMode>(m, "StepMode")
        .value("Conservative", StepMode::Conservative)
        .value("Aggressive", StepMode::Aggressive)
        ;

    py::class_<OptimumOutputOptions, OutputterOptions>(m, "OptimumOutputOptions")
        .def(py::init<>())
        .def_readwrite("xprefix", &OptimumOutputOptions::xprefix)
        .def_readwrite("yprefix", &OptimumOutputOptions::yprefix)
        .def_readwrite("zprefix", &OptimumOutputOptions::zprefix)
        .def_readwrite("xnames", &OptimumOutputOptions::xnames)
        .def_readwrite("ynames", &OptimumOutputOptions::ynames)
        .def_readwrite("znames", &OptimumOutputOptions::znames)
        ;

    py::class_<OptimumOptions>(m, "OptimumOptions")
        .def(py::init<>())
        .def_readwrite("output", &OptimumOptions::output)
        .def_readwrite("tolerance", &OptimumOptions::tolerance)
        .def_readwrite("tolerancex", &OptimumOptions::tolerancex)
        .def_readwrite("tolerancef", &OptimumOptions::tolerancef)
        .def_readwrite("max_iterations", &OptimumOptions::max_iterations)
        .def_readwrite("mu", &OptimumOptions::mu)
        .def_readwrite("tau", &OptimumOptions::tau)
        .def_readwrite("step", &OptimumOptions::step)
        .def_readwrite("kkt", &OptimumOptions::kkt)
        ;
}
