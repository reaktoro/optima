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
#include <pybind11/eigen.h>
namespace py = pybind11;

// Optima includes
#include <Optima/OptimumState.hpp>
using namespace Optima;

void exportOptimumState(py::module& m)
{
    const auto getH = [](const OptimumState& self) { return self.H; };
    const auto setH = [](OptimumState& self, VariantMatrixConstRef other) { return self.H = other; };

    py::class_<OptimumState>(m, "OptimumState")
        .def(py::init<>())
        .def_readwrite("x", &OptimumState::x)
        .def_readwrite("y", &OptimumState::y)
        .def_readwrite("z", &OptimumState::z)
        .def_readwrite("w", &OptimumState::w)
        .def_readwrite("f", &OptimumState::f)
        .def_readwrite("g", &OptimumState::g)
        .def_property("H", getH, setH)
        ;
}
