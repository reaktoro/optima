// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
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
#include "pybind11.hxx"

// Optima includes
#include <Optima/MasterMatrix.hpp>
#include <Optima/Stability.hpp>
using namespace Optima;

void exportStability(py::module& m)
{
    py::class_<StabilityStatus>(m, "StabilityStatus")
        .def_readonly("js"   , &StabilityStatus::js)
        .def_readonly("ju"   , &StabilityStatus::ju)
        .def_readonly("jlu"  , &StabilityStatus::jlu)
        .def_readonly("juu"  , &StabilityStatus::juu)
        .def_readonly("s"    , &StabilityStatus::s)
        ;

    auto update = [](Stability& self,
        MatrixView Wx,
        VectorView g,
        VectorView x,
        VectorView w,
        VectorView xlower,
        VectorView xupper,
        IndicesView jb)
    {
        self.update({Wx, g, x, w, xlower, xupper, jb});
    };

    py::class_<Stability>(m, "Stability")
        .def(py::init<>())
        .def(py::init<Index>())
        .def("update", update)
        .def("status", &Stability::status, PYBIND_ENSURE_MUTUAL_EXISTENCE)
        ;
}
