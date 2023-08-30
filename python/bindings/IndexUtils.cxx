// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2023 Allan Leal
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
#include <Optima/IndexUtils.hpp>
using namespace Optima;

void exportIndexUtils(py::module& m)
{
    m.def("indices", &indices);
    m.def("contains", &contains);
    m.def("moveIntersectionLeft", &moveIntersectionLeft);
    m.def("moveIntersectionRight", &moveIntersectionRight);
    m.def("moveIntersectionLeftStable", &moveIntersectionLeftStable);
    m.def("moveIntersectionRightStable", &moveIntersectionRightStable);
    m.def("difference", &difference);
    m.def("intersect", &intersect);
    m.def("isIntersectionEmpty", &isIntersectionEmpty);
}
