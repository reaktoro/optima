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
#include <Optima/HessianMatrix.hpp>
using namespace Optima;

void exportHessianMatrix(py::module& m)
{
    py::class_<HessianMatrixConstRef>(m, "HessianMatrixConstRef")
        .def(py::init<>())
        .def(py::init<VectorConstRef>()) // IMPORTANT: This constructor with VectorConstRef needs to come first than the one with MatrixConstRef, otherwise 1d numpy arrays will be interpreted by pydind11 as MatrixConstRef, instead of VectorConstRef.
        .def(py::init<MatrixConstRef>())
        .def(py::init<HessianMatrixRef>())
        .def(py::init<HessianMatrixConstRef>())
        .def_readonly("dense", &HessianMatrixConstRef::dense)
        .def_readonly("diagonal", &HessianMatrixConstRef::diagonal)
        .def_readonly("structure", &HessianMatrixConstRef::structure)
        ;

    py::class_<HessianMatrixRef>(m, "HessianMatrixRef")
        .def(py::init<MatrixRef, VectorRef>())
        .def(py::init<HessianMatrixRef>())
        .def_readwrite("dense", &HessianMatrixRef::dense)
        .def_readwrite("diagonal", &HessianMatrixRef::diagonal)
        .def_readonly("structure", &HessianMatrixRef::structure)
        ;
}
