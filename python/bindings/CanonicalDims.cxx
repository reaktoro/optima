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
#include <Optima/CanonicalDims.hpp>
using namespace Optima;

void exportCanonicalDims(py::module& m)
{
    py::class_<CanonicalDims>(m, "CanonicalDims")
        .def_readonly("nx" , &CanonicalDims::nx)
        .def_readonly("np" , &CanonicalDims::np)
        .def_readonly("ny" , &CanonicalDims::ny)
        .def_readonly("nz" , &CanonicalDims::nz)
        .def_readonly("nw" , &CanonicalDims::nw)
        .def_readonly("nt" , &CanonicalDims::nt)
        .def_readonly("ns" , &CanonicalDims::ns)
        .def_readonly("nu" , &CanonicalDims::nu)
        .def_readonly("nb" , &CanonicalDims::nb)
        .def_readonly("nn" , &CanonicalDims::nn)
        .def_readonly("nl" , &CanonicalDims::nl)
        .def_readonly("nbs", &CanonicalDims::nbs)
        .def_readonly("nbu", &CanonicalDims::nbu)
        .def_readonly("nns", &CanonicalDims::nns)
        .def_readonly("nnu", &CanonicalDims::nnu)
        .def_readonly("nbe", &CanonicalDims::nbe)
        .def_readonly("nbi", &CanonicalDims::nbi)
        .def_readonly("nne", &CanonicalDims::nne)
        .def_readonly("nni", &CanonicalDims::nni)
        ;
}
