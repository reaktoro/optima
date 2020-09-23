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
namespace py = pybind11;

// Optima includes
#include <Optima/Number.hpp>
using namespace Optima;

template<typename T>
void exportNumberAux(py::module& m, const char* type)
{
    py::class_<Number<T>>(m, type)
        .def(py::init<>())
        .def(py::init<T>())
        .def_readwrite("value", &Number<T>::value)
        ;
}

void exportNumber(py::module& m)
{
    exportNumberAux<double>(m, "DoubleNumber");
    exportNumberAux<Index>(m, "IndexNumber");
}
