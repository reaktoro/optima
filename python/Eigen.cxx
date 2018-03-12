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
#include <eigen3/Eigenx/Core>
#include <eigen3/Eigen/LU>
using namespace Eigen;

void exportEigen(py::module& m)
{
    auto eigen = m.def_submodule("eigen");

    eigen.def("ones", [](int n) -> VectorXd { return ones(n); });
    eigen.def("ones", [](int m, int n) -> MatrixXd { return ones(m, n); });
    eigen.def("zeros", [](int n) -> VectorXd { return zeros(n); });
    eigen.def("zeros", [](int m, int n) -> MatrixXd { return zeros(m, n); });
    eigen.def("random", [](int n) -> VectorXd { return random(n); });
    eigen.def("random", [](int m, int n) -> MatrixXd { return random(m, n); });
    eigen.def("eye", [](int n) -> MatrixXd { return identity(n, n); });
    eigen.def("diag", [](VectorXdConstRef x) -> MatrixXd { return x.asDiagonal(); });
    eigen.def("vector", [](int n=0) -> VectorXd { return VectorXd(n); }, py::arg("n")=0);
    eigen.def("matrix", [](int m=0, int n=0) -> MatrixXd { return MatrixXd(m, n); }, py::arg("m")=0, py::arg("n")=0);
    eigen.def("solve", [](MatrixXdConstRef A, VectorXdConstRef b) -> VectorXd { return A.fullPivLu().solve(b); });
}
