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
#include <Optima/LU.hpp>
using namespace Optima;

void exportLU(py::module& m)
{
    auto init = [](MatrixConstRef4py A) -> LU
    {
        return LU(A);
    };

    auto decompose = [](LU& self, MatrixConstRef4py A)
    {
        return self.decompose(A);
    };

    Matrix Xtmp;

    auto solve1 = [=](LU& self, MatrixConstRef4py B, MatrixRef4py X) mutable
    {
        Xtmp.resize(X.rows(), X.cols());
        self.solve(B, Xtmp);
        X = Xtmp;
    };

    auto solve2 = [=](LU& self, MatrixRef4py X) mutable
    {
        Xtmp = X;
        self.solve(Xtmp);
        X = Xtmp;
    };

    py::class_<LU>(m, "LU")
        .def(py::init<>())
        .def(py::init(init))
        .def("empty", &LU::empty)
        .def("decompose", decompose)
        .def("solve", solve1)
        .def("solve", solve2)
        .def("rank", &LU::rank)
        // .def("P", &LU::P)
        // .def("Q", &LU::Q)
        ;
}
