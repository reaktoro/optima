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
#include <Optima/MasterVector.hpp>
using namespace Optima;

void exportMasterVector(py::module& m)
{
    auto __add__      = [](const MasterVector& l, const MasterVector& r) -> MasterVector { return l + r; };
    auto __sub__      = [](const MasterVector& l, const MasterVector& r) -> MasterVector { return l - r; };
    auto __mul__      = [](const MasterVector& l, double r) -> MasterVector { return l * r; };
    auto __truediv__  = [](const MasterVector& l, double r) -> MasterVector { return l / r; };

    py::class_<MasterVector>(m, "MasterVector")
        .def(py::init<>())
        .def(py::init<const MasterDims&>())
        .def(py::init<Index, Index, Index>())
        .def(py::init<VectorView, Index, Index, Index>())
        .def(py::init<const MasterVector&>())
        .def(py::init<const MasterVectorRef&>())
        .def(py::init<const MasterVectorView&>())
        .def_property("x", [](MasterVector& s) -> VectorRef { return s.x; }, [](MasterVector& s, VectorView x) { s.x = x; })
        .def_property("p", [](MasterVector& s) -> VectorRef { return s.p; }, [](MasterVector& s, VectorView p) { s.p = p; })
        .def_property("w", [](MasterVector& s) -> VectorRef { return s.w; }, [](MasterVector& s, VectorView w) { s.w = w; })
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= double())
        .def(py::self /= double())
        .def("__add__", __add__)
        .def("__sub__", __sub__)
        .def("__mul__", __mul__)
        .def("__rmul__", __mul__)
        .def("__truediv__", __truediv__)
        .def("resize", &MasterVector::resize)
        .def("dot", [](const MasterVector& l, const MasterVector& r) { return l.dot(r); })
        .def("norm", &MasterVector::norm)
        .def("squaredNorm", &MasterVector::squaredNorm)
        .def("size", &MasterVector::size)
        .def("array", [](const MasterVector& self) { return Vector(self); })
        .def(double() * py::self)
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

    py::class_<MasterVectorView>(m, "MasterVectorView")
        .def(py::init<VectorView, Index, Index, Index>())
        .def(py::init<const MasterVector&>())
        .def(py::init<const MasterVectorRef&>())
        .def_readonly("x", &MasterVectorView::x)
        .def_readonly("p", &MasterVectorView::p)
        .def_readonly("w", &MasterVectorView::w)
        .def("size", &MasterVectorView::size)
        .def("array", [](const MasterVectorView& self) { return Vector(self); })
        ;

    py::implicitly_convertible<MasterVector, MasterVectorRef>();
    py::implicitly_convertible<MasterVector, MasterVectorView>();
    py::implicitly_convertible<MasterVectorRef, MasterVectorView>();
}
