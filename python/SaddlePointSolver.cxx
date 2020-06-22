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
#include <Optima/Result.hpp>
#include <Optima/SaddlePointOptions.hpp>
#include <Optima/SaddlePointSolver.hpp>
using namespace Optima;

void exportSaddlePointSolver(py::module& m)
{
    auto init = [](Index n, Index m, MatrixConstRef4py A) -> SaddlePointSolver
    {
        return SaddlePointSolver({ n, m, A });
    };

    auto canonicalize = [](SaddlePointSolver& self, MatrixConstRef4py H, MatrixConstRef4py J, IndicesConstRef jf)
    {
        return self.canonicalize({ H, J, jf });
    };

    auto decompose = [](SaddlePointSolver& self, MatrixConstRef4py H, MatrixConstRef4py J, IndicesConstRef jf)
    {
        return self.decompose({ H, J, jf });
    };

    auto solve1 = [](SaddlePointSolver& self, VectorConstRef a, VectorConstRef b, VectorRef x, VectorRef y)
    {
        self.solve({ a, b, x, y });
    };

    auto solve2 = [](SaddlePointSolver& self, VectorRef x, VectorRef y)
    {
        self.solve({ x, y });
    };

    auto solve3 = [](SaddlePointSolver& self, MatrixConstRef4py H, MatrixConstRef4py J, VectorConstRef x, VectorConstRef g, VectorConstRef b, VectorConstRef h, VectorRef xbar, VectorRef ybar)
    {
        self.solve({ H, J, x, g, b, h, xbar, ybar });
    };

    auto residuals1 = [](SaddlePointSolver& self, VectorConstRef x, VectorConstRef b, VectorRef r, VectorRef e)
    {
        self.residuals({ x, b, r, e });
    };

    auto residuals2 = [](SaddlePointSolver& self, MatrixConstRef4py J, VectorConstRef x, VectorConstRef b, VectorConstRef h, VectorRef r, VectorRef e)
    {
        self.residuals({ J, x, b, h, r, e });
    };

    py::class_<SaddlePointSolverInfo>(m, "SaddlePointSolverInfo")
        .def_readonly("jb", &SaddlePointSolverInfo::jb)
        .def_readonly("jn", &SaddlePointSolverInfo::jn)
        .def_readonly("R", &SaddlePointSolverInfo::R)
        .def_readonly("S", &SaddlePointSolverInfo::S)
        .def_readonly("Q", &SaddlePointSolverInfo::Q)
        ;

    py::class_<SaddlePointSolver>(m, "SaddlePointSolver")
        .def(py::init(init))
        .def("setOptions", &SaddlePointSolver::setOptions)
        .def("options", &SaddlePointSolver::options)
        .def("canonicalize", canonicalize)
        .def("decompose", decompose)
        .def("solve", solve1)
        .def("solve", solve2)
        .def("solve", solve3)
        .def("residuals", residuals1)
        .def("residuals", residuals2)
        .def("info", &SaddlePointSolver::info, py::return_value_policy::reference_internal)
        ;
}
