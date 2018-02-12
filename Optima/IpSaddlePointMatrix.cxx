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
#include <Optima/IpSaddlePointMatrix.hpp>
using namespace Optima;

void exportIpSaddlePointMatrix(py::module& m)
{
    py::class_<IpSaddlePointMatrix>(m, "IpSaddlePointMatrix")
        .def(py::init<MatrixXdConstRef, MatrixXdConstRef, VectorXdConstRef, VectorXdConstRef, VectorXdConstRef, VectorXdConstRef, Index, Index>(), py::arg("H"), py::arg("A"), py::arg("Z"), py::arg("W"), py::arg("L"), py::arg("U"), py::arg("nx"), py::arg("nf"))
        .def("H", &IpSaddlePointMatrix::H, py::return_value_policy::reference_internal, "Return the Hessian matrix `H` in the saddle point matrix.")
        .def("A", &IpSaddlePointMatrix::A, py::return_value_policy::reference_internal, "Return the Jacobian matrix `A` in the saddle point matrix.")
        .def("Z", &IpSaddlePointMatrix::Z, py::return_value_policy::reference_internal, "Return the diagonal matrix `Z` in the saddle point matrix.")
        .def("W", &IpSaddlePointMatrix::W, py::return_value_policy::reference_internal, "Return the diagonal matrix `W` in the saddle point matrix.")
        .def("L", &IpSaddlePointMatrix::L, py::return_value_policy::reference_internal, "Return the diagonal matrix `L` in the saddle point matrix.")
        .def("U", &IpSaddlePointMatrix::U, py::return_value_policy::reference_internal, "Return the diagonal matrix `U` in the saddle point matrix.")
        .def("nx", &IpSaddlePointMatrix::nx, "Return the number of free variables.")
        .def("nf", &IpSaddlePointMatrix::nf, "Return the number of fixed variables.")
        .def("n", &IpSaddlePointMatrix::n, "Return the number of variables.")
        .def("m", &IpSaddlePointMatrix::m, "Return the number of linear equality constraints.")
        .def("size", &IpSaddlePointMatrix::size, "Return the dimension of the saddle point matrix.")
        .def("matrix", &IpSaddlePointMatrix::matrix, "Convert this IpSaddlePointMatrix instance into a numpy.ndarray instance.")
        ;

    py::class_<IpSaddlePointVector>(m, "IpSaddlePointVector")
        .def(py::init<VectorXdConstRef, VectorXdConstRef, VectorXdConstRef, VectorXdConstRef>(), py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"))
        .def(py::init<VectorXdConstRef, Index, Index>(), py::arg("r"), py::arg("n"), py::arg("m"))
        .def("size", &IpSaddlePointVector::size, "Return the dimension of the saddle point vector.")
        .def("a", &IpSaddlePointVector::a, py::return_value_policy::reference_internal, "Return the solution vector *a*.")
        .def("b", &IpSaddlePointVector::b, py::return_value_policy::reference_internal, "Return the solution vector *b*.")
        .def("c", &IpSaddlePointVector::c, py::return_value_policy::reference_internal, "Return the solution vector *c*.")
        .def("d", &IpSaddlePointVector::d, py::return_value_policy::reference_internal, "Return the solution vector *d*.")
        .def("vector", &IpSaddlePointVector::vector, "Convert this IpSaddlePointVector instance into a numpy.ndarray instance.")
        ;

    py::class_<IpSaddlePointSolution>(m, "IpSaddlePointSolution")
        .def(py::init<VectorXdRef, VectorXdRef, VectorXdRef, VectorXdRef>(), py::arg("x"), py::arg("y"), py::arg("z"), py::arg("w"))
        .def(py::init<VectorXdRef, Index, Index>(), py::arg("s"), py::arg("n"), py::arg("m"))
        .def("size", &IpSaddlePointSolution::size, "Return the dimension of the saddle point solution vector.")
        .def("x", &IpSaddlePointSolution::x, "Return the solution vector *x*.")
        .def("y", &IpSaddlePointSolution::y, "Return the solution vector *y*.")
        .def("z", &IpSaddlePointSolution::z, "Return the solution vector *z*.")
        .def("w", &IpSaddlePointSolution::w, "Return the solution vector *w*.")
        .def("vector", &IpSaddlePointSolution::vector, "Convert this IpSaddlePointSolution instance into a numpy.ndarray instance.")
        ;
}
