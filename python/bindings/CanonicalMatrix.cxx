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
#include <Optima/CanonicalMatrix.hpp>
using namespace Optima;

void exportCanonicalMatrix(py::module& m)
{
    py::class_<CanonicalMatrix>(m, "CanonicalMatrix")
        .def(py::init<CanonicalMatrix const&>())
        .def_readonly("dims" , &CanonicalMatrix::dims)
        .def_readonly("Hss"  , &CanonicalMatrix::Hss)
        .def_readonly("Hsp"  , &CanonicalMatrix::Hsp)
        .def_readonly("Vps"  , &CanonicalMatrix::Vps)
        .def_readonly("Vpp"  , &CanonicalMatrix::Vpp)
        .def_readonly("Sbsns", &CanonicalMatrix::Sbsns)
        .def_readonly("Sbsp" , &CanonicalMatrix::Sbsp)
        .def_readonly("Rbs"  , &CanonicalMatrix::Rbs)
        .def_readonly("jb"   , &CanonicalMatrix::jb)
        .def_readonly("jn"   , &CanonicalMatrix::jn)
        .def_readonly("js"   , &CanonicalMatrix::js)
        .def_readonly("ju"   , &CanonicalMatrix::ju)
        ;
}
