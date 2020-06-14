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
#include <Optima/SaddlePointSolverLegacy.hpp>
using namespace Optima;

void exportSaddlePointSolverLegacy(py::module& m)
{
    auto init = [](Index n, Index m, MatrixConstRef4py A) -> SaddlePointSolverLegacy
    {
        return SaddlePointSolverLegacy({ n, m, A });
    };

    auto canonicalize = [](SaddlePointSolverLegacy& self, MatrixConstRef4py H, MatrixConstRef4py J, IndicesConstRef ifixed)
    {
        return self.canonicalize({ H, J, ifixed });
    };

    auto decompose = [](SaddlePointSolverLegacy& self, MatrixConstRef4py H, MatrixConstRef4py J, MatrixConstRef4py G, IndicesConstRef ifixed)
    {
        return self.decompose({ H, J, G, ifixed });
    };

    auto solve1 = [](SaddlePointSolverLegacy& self, VectorConstRef a, VectorConstRef b, VectorRef x, VectorRef y)
    {
        self.solve({ a, b, x, y });
    };

    auto solve2 = [](SaddlePointSolverLegacy& self, VectorRef x, VectorRef y)
    {
        self.solve({ x, y });
    };

    auto solve3 = [](SaddlePointSolverLegacy& self, MatrixConstRef4py H, MatrixConstRef4py J, VectorConstRef x, VectorConstRef g, VectorConstRef b, VectorConstRef h, VectorRef xbar, VectorRef ybar)
    {
        self.solve({ H, J, x, g, b, h, xbar, ybar });
    };

    auto residuals1 = [](SaddlePointSolverLegacy& self, VectorConstRef x, VectorConstRef b, VectorRef r)
    {
        self.residuals({ x, b, r });
    };

    auto residuals2 = [](SaddlePointSolverLegacy& self, MatrixConstRef4py J, VectorConstRef x, VectorConstRef b, VectorConstRef h, VectorRef r)
    {
        self.residuals({ J, x, b, h, r });
    };

    py::class_<SaddlePointSolverLegacyInfo>(m, "SaddlePointSolverLegacyInfo")
        .def_readonly("jb", &SaddlePointSolverLegacyInfo::jb)
        .def_readonly("jn", &SaddlePointSolverLegacyInfo::jn)
        .def_readonly("R", &SaddlePointSolverLegacyInfo::R)
        .def_readonly("S", &SaddlePointSolverLegacyInfo::S)
        .def_readonly("Q", &SaddlePointSolverLegacyInfo::Q)
        ;

    py::class_<SaddlePointSolverLegacy>(m, "SaddlePointSolverLegacy")
        .def(py::init<>())
        .def(py::init(init))
        .def("setOptions", &SaddlePointSolverLegacy::setOptions)
        .def("options", &SaddlePointSolverLegacy::options)
        .def("canonicalize", canonicalize)
        .def("decompose", decompose)
        .def("solve", solve1)
        .def("solve", solve2)
        .def("solve", solve3)
        .def("residuals", residuals1)
        .def("residuals", residuals2)
        .def("info", &SaddlePointSolverLegacy::info, py::return_value_policy::reference_internal)
        ;
}
