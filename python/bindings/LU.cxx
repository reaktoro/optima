// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2020 Allan Leal
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
    auto decompose = [](LU& self, MatrixView4py A)
    {
        return self.decompose(A);
    };

    auto solve1 = [=](LU& self, VectorView b, VectorRef x) mutable
    {
        self.solve(b, x);
    };

    auto solve2 = [=](LU& self, VectorRef x) mutable
    {
        self.solve(x);
    };

    auto P = [=](LU& self) -> Indices
    {
        self.P().indices();
    };

    auto Q = [=](LU& self) -> Indices
    {
        self.Q().indices();
    };

    py::class_<LU>(m, "LU")
        .def(py::init<>())
        .def("empty", &LU::empty)
        .def("decompose", decompose)
        .def("solve", solve1)
        .def("solve", solve2)
        .def("rank", &LU::rank)
        .def("P", P)
        .def("Q", Q)
        ;
}
