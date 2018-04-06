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
#include <Optima/SaddlePointMatrix.hpp>
using namespace Optima;

void exportSaddlePointMatrix(py::module& m)
{
    py::class_<SaddlePointMatrix>(m, "SaddlePointMatrix")
        .def_readonly("H", &SaddlePointMatrix::H)
        .def_readonly("D", &SaddlePointMatrix::D)
        .def_readonly("A", &SaddlePointMatrix::A)
        .def_readonly("G", &SaddlePointMatrix::G)
        .def_readonly("jf", &SaddlePointMatrix::jf)
        .def(py::init<VariantMatrixConstRef, VectorConstRef, MatrixConstRef, IndicesConstRef>(), py::arg("H"), py::arg("D"), py::arg("A"), py::arg("jf"))
        .def(py::init<VariantMatrixConstRef, VectorConstRef, MatrixConstRef, MatrixConstRef, IndicesConstRef>(), py::arg("H"), py::arg("D"), py::arg("A"), py::arg("G"), py::arg("jf"))

        // IMPORTANT: Constructors below are needed to allow 1d numpy arrays to be converted to VariantMatrixConstRef
        .def(py::init<VectorConstRef, VectorConstRef, MatrixConstRef, IndicesConstRef>(), py::arg("H"), py::arg("D"), py::arg("A"), py::arg("jf"))
        .def(py::init<VectorConstRef, VectorConstRef, MatrixConstRef, MatrixConstRef, IndicesConstRef>(), py::arg("H"), py::arg("D"), py::arg("A"), py::arg("G"), py::arg("jf"))

        // IMPORTANT: Constructors below are needed to allow 2d numpy arrays to be converted to VariantMatrixConstRef
        .def(py::init<MatrixConstRef, VectorConstRef, MatrixConstRef, IndicesConstRef>(), py::arg("H"), py::arg("D"), py::arg("A"), py::arg("jf"))
        .def(py::init<MatrixConstRef, VectorConstRef, MatrixConstRef, MatrixConstRef, IndicesConstRef>(), py::arg("H"), py::arg("D"), py::arg("A"), py::arg("G"), py::arg("jf"))
        .def("array", [](SaddlePointMatrix self) { return Matrix(self); })
        ;

    py::class_<SaddlePointVector>(m, "SaddlePointVector")
         .def_readonly("x", &SaddlePointVector::x)
         .def_readonly("y", &SaddlePointVector::y)
         .def_readonly("a", &SaddlePointVector::a)
         .def_readonly("b", &SaddlePointVector::b)
         .def(py::init<VectorConstRef, VectorConstRef>(), py::arg("x"), py::arg("y"))
         .def(py::init<VectorConstRef, Index, Index>(), py::arg("r"), py::arg("n"), py::arg("m"))
         .def("array", [](SaddlePointVector self) { return Vector(self); })
         ;

     py::class_<SaddlePointSolution>(m, "SaddlePointSolution")
         .def_readwrite("x", &SaddlePointSolution::x)
         .def_readwrite("y", &SaddlePointSolution::y)
         .def(py::init<VectorRef, VectorRef>(), py::arg("x"), py::arg("y"))
         .def(py::init<VectorRef, Index, Index>(), py::arg("s"), py::arg("n"), py::arg("m"))
         .def("array", [](SaddlePointSolution self) { return Vector(self); })
         ;
}
