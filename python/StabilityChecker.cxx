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
#include <Optima/StabilityChecker.hpp>
#include <Optima/Options.hpp>
using namespace Optima;

void exportStabilityChecker(py::module& m)
{
    auto init = [](Index n, Index m, MatrixConstRef4py A) -> StabilityChecker
    {
        return StabilityChecker({n, m, A});
    };

    auto initialize = [](StabilityChecker& self,
        MatrixConstRef4py A,
        VectorConstRef b,
        VectorConstRef xlower,
        VectorConstRef xupper)
    {
        self.initialize({ A, b, xlower, xupper });
    };

    auto update = [](StabilityChecker& self,
        MatrixConstRef W,
        VectorConstRef x,
        VectorConstRef y,
        VectorConstRef g,
        VectorConstRef xlower,
        VectorConstRef xupper)
    {
        self.update({ W, x, y, g, xlower, xupper });
    };

    py::class_<StabilityChecker>(m, "StabilityChecker")
        .def(py::init<>())
        .def(py::init(init))
        .def("initialize", initialize)
        .def("update", update)
        .def("stability", &StabilityChecker::stability, py::return_value_policy::reference_internal)
        ;
}
