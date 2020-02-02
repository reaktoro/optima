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
        return SaddlePointSolver({n, m, A});
    };

    auto decompose = [](SaddlePointSolver& self, MatrixConstRef4py H, MatrixConstRef4py J, MatrixConstRef4py G, IndicesConstRef ifixed)
    {
        return self.decompose({H, J, G, ifixed});
    };

    auto solve1 = [](SaddlePointSolver& self, VectorConstRef a, VectorConstRef b, VectorRef x, VectorRef y)
    {
        self.solve({a, b, x, y});
    };

    auto solve2 = [](SaddlePointSolver& self, VectorRef x, VectorRef y)
    {
        self.solve({x, y});
    };

    py::class_<SaddlePointSolver>(m, "SaddlePointSolver")
        .def(py::init<>())
        .def(py::init(init))
        .def("setOptions", &SaddlePointSolver::setOptions)
        .def("options", &SaddlePointSolver::options)
        .def("decompose", decompose)
        .def("solve", solve1)
        .def("solve", solve2)
        ;
}
