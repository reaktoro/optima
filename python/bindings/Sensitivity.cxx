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
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

// Optima includes
#include <Optima/Sensitivity.hpp>
using namespace Optima;

void exportSensitivity(py::module& m)
{
    py::class_<Sensitivity>(m, "Sensitivity")
        .def(py::init<>())
        .def(py::init<const Dims&, Index>())
        .def_readonly("dims", &Sensitivity::dims)
        .def_readwrite("dxdc", &Sensitivity::dxdc)
        .def_readwrite("dpdc", &Sensitivity::dpdc)
        .def_readwrite("dyedc", &Sensitivity::dyedc)
        .def_readwrite("dygdc", &Sensitivity::dygdc)
        .def_readwrite("dzedc", &Sensitivity::dzedc)
        .def_readwrite("dzgdc", &Sensitivity::dzgdc)
        .def_readwrite("dsdc", &Sensitivity::dsdc)
        .def_readwrite("dxbgdc", &Sensitivity::dxbgdc)
        .def_readwrite("dxhgdc", &Sensitivity::dxhgdc)
        ;
}
