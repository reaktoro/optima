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
#include <Optima/ActiveStepper.hpp>
#include <Optima/Options.hpp>
#include <Optima/SaddlePointMatrix.hpp>
using namespace Optima;

void exportActiveStepper(py::module& m)
{
    auto init = [](Index n, Index m, MatrixConstRef A, VectorConstRef xlower, VectorConstRef xupper, IndicesConstRef ilower, IndicesConstRef iupper, IndicesConstRef ifixed) -> ActiveStepper
    {
        return ActiveStepper({n, m, A, xlower, xupper, ilower, iupper, ifixed});
    };

    auto decompose = [](ActiveStepper& self, VectorConstRef x, VectorConstRef y, MatrixConstRef J, VectorConstRef g, MatrixConstRef H)
    {
        return self.decompose({x, y, J, g, H});
    };

    auto solve = [](ActiveStepper& self,  VectorConstRef x, VectorConstRef y, VectorConstRef b, VectorConstRef h, VectorConstRef g, VectorRef dx, VectorRef dy, VectorRef rx, VectorRef ry, VectorRef z)
    {
        self.solve({x, y, b, h, g}, {dx, dy, rx, ry, z});
    };

    py::class_<ActiveStepper>(m, "ActiveStepper")
        .def(py::init<>())
        .def(py::init(init))
        .def("setOptions", &ActiveStepper::setOptions)
        .def("decompose", decompose)
        .def("solve", solve)
        ;
}
