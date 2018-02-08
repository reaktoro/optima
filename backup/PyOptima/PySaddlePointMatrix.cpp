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
#include <Optima/Core/SaddlePointMatrix.hpp>
using namespace Optima;

void exportSaddlePointMatrix(py::module& m)
{
    py::class_<SaddlePointMatrix>(m, "SaddlePointMatrix")
        .def(py::init<MatrixXdConstRef, MatrixXdConstRef, MatrixXdConstRef, Index, Index>(),
            py::arg("H"), py::arg("A"), py::arg("G"), py::arg("nx"), py::arg("nf") = 0)
        .def("H", &SaddlePointMatrix::H, "Return the Hessian matrix *H*.")
        .def("A", &SaddlePointMatrix::A, "Return the Jacobian matrix *A*.")
        .def("G", &SaddlePointMatrix::G, "Return the matrix *G*.")
        .def("size", &SaddlePointMatrix::size, "Return the dimension of the saddle point matrix.")
        .def("n", &SaddlePointMatrix::n, "Return the number of variables.")
        .def("m", &SaddlePointMatrix::m, "Return the number of linear equality constraints.")
        .def("nx", &SaddlePointMatrix::nx, "Return the number of free variables.")
        .def("nf", &SaddlePointMatrix::nf, "Return the number of fixed variables.")
        .def("matrix", &SaddlePointMatrix::matrix, "Convert this SaddlePointMatrix instance into a Matrix instance.")
        ;
}
