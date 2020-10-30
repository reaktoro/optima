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
#include <Optima/MasterVector.hpp>
using namespace Optima;

void exportMasterVector(py::module& m)
{
    py::class_<CanonicalVector>(m, "CanonicalVector")
        .def_readonly("xs", &CanonicalVector::xs)
        .def_readonly("p", &CanonicalVector::p)
        .def_readonly("wbs", &CanonicalVector::wbs)
        ;

    py::class_<CanonicalVectorRef>(m, "CanonicalVectorRef")
        .def_readwrite("xs", &CanonicalVectorRef::xs)
        .def_readwrite("p", &CanonicalVectorRef::p)
        .def_readwrite("wbs", &CanonicalVectorRef::wbs)
        ;

    py::class_<MasterVector>(m, "MasterVector")
        .def(py::init<Index, Index, Index, Index>())
        .def(py::init<const MasterVector&>())
        .def_readwrite("x", &MasterVector::x)
        .def_readwrite("p", &MasterVector::p)
        .def_readwrite("y", &MasterVector::y)
        .def_readwrite("z", &MasterVector::z)
        .def_readwrite("w", &MasterVector::w)
        .def_readwrite("data", &MasterVector::data)
        ;
}
