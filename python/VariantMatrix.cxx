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
#include <Optima/VariantMatrix.hpp>
using namespace Optima;

void exportVariantMatrix(py::module& m)
{
    py::class_<VariantMatrix>(m, "VariantMatrix")
        .def(py::init<>())
        .def(py::init<VariantMatrixConstRef>())
        .def_readwrite("dense", &VariantMatrix::dense)
        .def_readwrite("diagonal", &VariantMatrix::diagonal)
        .def_readwrite("structure", &VariantMatrix::structure)
        .def("setZero", &VariantMatrix::setZero)
        .def("setDense", &VariantMatrix::setDense)
        .def("setDiagonal", &VariantMatrix::setDiagonal)
        ;

    py::class_<VariantMatrixRef>(m, "VariantMatrixRef")
        .def(py::init<VariantMatrix&>())
        .def_readwrite("dense", &VariantMatrixRef::dense)
        .def_readwrite("diagonal", &VariantMatrixRef::diagonal)
        .def_property_readonly("structure", [](const VariantMatrixRef& self) { return self.structure; })
        ;

    py::class_<VariantMatrixConstRef>(m, "VariantMatrixConstRef")
        .def(py::init<>())
        .def(py::init<VectorConstRef>()) // IMPORTANT: This constructor with VectorConstRef needs to come first than the one with MatrixConstRef, otherwise 1d numpy arrays will be interpreted by pydind11 as MatrixConstRef, instead of VectorConstRef.
        .def(py::init<MatrixConstRef>())
        .def(py::init<VariantMatrixRef>())
        .def(py::init<const VariantMatrix&>())
        .def_readonly("dense", &VariantMatrixConstRef::dense)
        .def_readonly("diagonal", &VariantMatrixConstRef::diagonal)
        .def_property_readonly("structure", [](const VariantMatrixConstRef& self) { return self.structure; })
        ;

    py::implicitly_convertible<py::array, VariantMatrixConstRef>();
}
