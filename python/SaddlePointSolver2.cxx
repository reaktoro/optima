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
#include <Optima/SaddlePointSolver2.hpp>
using namespace Optima;

void exportSaddlePointSolver2(py::module& m)
{
    auto init = [](Index n, Index m, MatrixConstRef4py A) -> SaddlePointSolver2
    {
        return SaddlePointSolver2({n, m, A});
    };

    auto decompose = [](SaddlePointSolver2& self,
        MatrixConstRef4py H,
        MatrixConstRef4py J,
        MatrixConstRef4py G,
        VectorConstRef D,
        VectorConstRef V,
        IndicesConstRef ifixed,
        IndicesConstRef idiagonal)
    {
        return self.decompose({ H, J, G, D, V, ifixed, idiagonal });
    };

    auto solve1 = [](SaddlePointSolver2& self,
        VectorConstRef a,
        VectorConstRef b,
        VectorRef x,
        VectorRef y)
    {
        self.solve({ a, b, x, y });
    };

    auto solve2 = [](SaddlePointSolver2& self, VectorRef x, VectorRef y)
    {
        self.solve({ x, y });
    };

    py::class_<SaddlePointSolver2>(m, "SaddlePointSolver2")
        .def(py::init<>())
        .def(py::init(init))
        .def("setOptions", &SaddlePointSolver2::setOptions)
        .def("options", &SaddlePointSolver2::options)
        .def("decompose", decompose)
        .def("solve", solve1)
        .def("solve", solve2)
        ;
}
