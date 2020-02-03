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
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
namespace py = pybind11;

// Optima includes
#include <Optima/Problem.hpp>
using namespace Optima;

void exportProblem(py::module& m)
{
    py::class_<Problem>(m, "Problem")
        .def(py::init<const Dims&>())
        .def_readonly("dims", &Problem::dims)
        .def_readwrite("Ae", &Problem::Ae)
        .def_readwrite("Ag", &Problem::Ag)
        .def_readwrite("be", &Problem::be)
        .def_readwrite("bg", &Problem::bg)
        .def_readwrite("he", &Problem::__4py_he)
        .def_readwrite("hg", &Problem::__4py_hg)
        .def_readwrite("f", &Problem::__4py_f)
        .def_readwrite("xlower", &Problem::xlower)
        .def_readwrite("xupper", &Problem::xupper)
        .def_readwrite("dgdp", &Problem::dgdp)
        .def_readwrite("dhdp", &Problem::dhdp)
        .def_readwrite("dbdp", &Problem::dbdp)
        ;
}
