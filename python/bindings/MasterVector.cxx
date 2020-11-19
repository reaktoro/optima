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
    py::class_<MasterVector>(m, "MasterVector")
        .def(py::init<const MasterDims&>())
        .def(py::init<Index, Index, Index>())
        .def(py::init<VectorConstRef, Index, Index, Index>())
        .def(py::init<const MasterVector&>())
        .def(py::init<const MasterVectorRef&>())
        .def(py::init<const MasterVectorConstRef&>())
        .def_property("x", [](MasterVector& s) -> VectorRef { return s.x; }, [](MasterVector& s, VectorConstRef x) { s.x = x; })
        .def_property("p", [](MasterVector& s) -> VectorRef { return s.p; }, [](MasterVector& s, VectorConstRef p) { s.p = p; })
        .def_property("w", [](MasterVector& s) -> VectorRef { return s.w; }, [](MasterVector& s, VectorConstRef w) { s.w = w; })
        .def("size", &MasterVector::size)
        .def("array", [](const MasterVector& self) { return Vector(self); })
        ;

    py::class_<MasterVectorRef>(m, "MasterVectorRef")
        .def(py::init<VectorRef, Index, Index, Index>())
        .def(py::init<MasterVector&>())
        .def(py::init<MasterVectorRef&>())
        .def_readwrite("x", &MasterVectorRef::x)
        .def_readwrite("p", &MasterVectorRef::p)
        .def_readwrite("w", &MasterVectorRef::w)
        .def("size", &MasterVectorRef::size)
        .def("array", [](const MasterVectorRef& self) { return Vector(self); })
        ;

    py::class_<MasterVectorConstRef>(m, "MasterVectorConstRef")
        .def(py::init<VectorConstRef, Index, Index, Index>())
        .def(py::init<const MasterVector&>())
        .def(py::init<const MasterVectorRef&>())
        .def_readonly("x", &MasterVectorConstRef::x)
        .def_readonly("p", &MasterVectorConstRef::p)
        .def_readonly("w", &MasterVectorConstRef::w)
        .def("size", &MasterVectorConstRef::size)
        .def("array", [](const MasterVectorConstRef& self) { return Vector(self); })
        ;

    py::implicitly_convertible<MasterVector, MasterVectorRef>();
    py::implicitly_convertible<MasterVector, MasterVectorConstRef>();
    py::implicitly_convertible<MasterVectorRef, MasterVectorConstRef>();
}
