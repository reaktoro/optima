// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
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
#include "pybind11.hxx"

// Optima includes
#include <Optima/CanonicalVector.hpp>
using namespace Optima;

void exportCanonicalVector(py::module& m)
{
    py::class_<CanonicalVector>(m, "CanonicalVector")
        .def(py::init<Index, Index, Index, Index>())
        .def(py::init<VectorView, Index, Index, Index, Index>())
        .def(py::init<const CanonicalVector&>())
        .def(py::init<const CanonicalVectorRef&>())
        .def(py::init<const CanonicalVectorView&>())
        .def_readwrite("xs" , &CanonicalVector::xs)
        .def_readwrite("xu" , &CanonicalVector::xu)
        .def_readwrite("p"  , &CanonicalVector::p)
        .def_readwrite("wbs", &CanonicalVector::wbs)
        .def("size", &CanonicalVector::size)
        .def("array", [](const CanonicalVector& self) { return Vector(self); })
        ;

    py::class_<CanonicalVectorRef>(m, "CanonicalVectorRef")
        .def(py::init<VectorRef, Index, Index, Index, Index>())
        .def(py::init<CanonicalVector&>())
        .def(py::init<CanonicalVectorRef&>())
        .def_readwrite("xs" , &CanonicalVectorRef::xs)
        .def_readwrite("xu" , &CanonicalVectorRef::xu)
        .def_readwrite("p"  , &CanonicalVectorRef::p)
        .def_readwrite("wbs", &CanonicalVectorRef::wbs)
        .def("size", &CanonicalVectorRef::size)
        .def("array", [](const CanonicalVectorRef& self) { return Vector(self); })
        ;

    py::class_<CanonicalVectorView>(m, "CanonicalVectorView")
        .def(py::init<VectorView, Index, Index, Index, Index>())
        .def(py::init<const CanonicalVector&>())
        .def(py::init<const CanonicalVectorRef&>())
        .def_readonly("xs" , &CanonicalVectorView::xs)
        .def_readonly("xu" , &CanonicalVectorView::xu)
        .def_readonly("p"  , &CanonicalVectorView::p)
        .def_readonly("wbs", &CanonicalVectorView::wbs)
        .def("size", &CanonicalVectorView::size)
        .def("array", [](const CanonicalVectorView& self) { return Vector(self); })
        ;

    py::implicitly_convertible<CanonicalVector, CanonicalVectorRef>();
    py::implicitly_convertible<CanonicalVector, CanonicalVectorView>();
    py::implicitly_convertible<CanonicalVectorRef, CanonicalVectorView>();
}
