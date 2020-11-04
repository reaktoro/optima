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
#include <Optima/MasterMatrix.hpp>
#include <Optima/Stability2.hpp>
using namespace Optima;

void exportStability2(py::module& m)
{
    py::class_<StabilityStatus>(m, "StabilityStatus")
        .def_readonly("js"   , &StabilityStatus::js)
        .def_readonly("ju"   , &StabilityStatus::ju)
        .def_readonly("jlu"  , &StabilityStatus::jlu)
        .def_readonly("juu"  , &StabilityStatus::juu)
        .def_readonly("s"    , &StabilityStatus::s)
        .def_readonly("lmbda", &StabilityStatus::lmbda)
        ;

    auto update = [](Stability2& self,
        MatrixViewRWQ RWQ,
        VectorConstRef g,
        VectorConstRef x,
        VectorConstRef xlower,
        VectorConstRef xupper)
    {
        self.update({RWQ, g, x, xlower, xupper});
    };

    py::class_<Stability2>(m, "Stability2")
        .def(py::init<Index>())
        .def("update", update)
        .def("status", &Stability2::status)
        ;
}
