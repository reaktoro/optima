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
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

// Optima includes
#include <Optima/OptimumParams.hpp>
using namespace Optima;

void exportOptimumParams(py::module& m)
{
    py::class_<OptimumParams>(m, "OptimumParams")
        .def(py::init<>())
        .def_readwrite("b", &OptimumParams::b)
        .def_readwrite("xlower", &OptimumParams::xlower)
        .def_readwrite("xupper", &OptimumParams::xupper)
        .def_readwrite("xfixed", &OptimumParams::xfixed)
        .def_readwrite("objective", &OptimumParams::objective)
        ;
}
