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
#include <Optima/Matrix.hpp>
#include <Eigen/LU>
using namespace Optima;

void exportEigen(py::module& m)
{
    auto eigen = m.def_submodule("eigen");

    eigen.def("ones", [](int n) -> Vector { return Eigen::ones(n); });
    eigen.def("ones", [](int m, int n) -> Matrix { return Eigen::ones(m, n); });
    eigen.def("zeros", [](int n) -> Vector { return Eigen::zeros(n); });
    eigen.def("zeros", [](int m, int n) -> Matrix { return Eigen::zeros(m, n); });
    eigen.def("random", [](int n) -> Vector { return Eigen::random(n); });
    eigen.def("random", [](int m, int n) -> Matrix { return Eigen::random(m, n); });
    eigen.def("eye", [](int n) -> Matrix { return Eigen::identity(n, n); });
    eigen.def("diag", [](VectorConstRef x) -> Matrix { return x.asDiagonal(); });
    eigen.def("vector", [](int n=0) -> Vector { return Vector(n); }, py::arg("n")=0);
    eigen.def("matrix", [](int m=0, int n=0) -> Matrix { return Matrix(m, n); }, py::arg("m")=0, py::arg("n")=0);
    eigen.def("solve", [](MatrixConstRef A, VectorConstRef b) -> Vector { return A.fullPivLu().solve(b); });
}
