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
#include <Optima/CanonicalMatrix.hpp>
using namespace Optima;

void exportCanonicalMatrix(py::module& m)
{
    py::class_<CanonicalDims>(m, "CanonicalDims")
        .def(py::init<>())
        .def_readwrite("nx" , &CanonicalDims::nx)
        .def_readwrite("np" , &CanonicalDims::np)
        .def_readwrite("ny" , &CanonicalDims::ny)
        .def_readwrite("nz" , &CanonicalDims::nz)
        .def_readwrite("nw" , &CanonicalDims::nw)
        .def_readwrite("ns" , &CanonicalDims::ns)
        .def_readwrite("nu" , &CanonicalDims::nu)
        .def_readwrite("nb" , &CanonicalDims::nb)
        .def_readwrite("nn" , &CanonicalDims::nn)
        .def_readwrite("nl" , &CanonicalDims::nl)
        .def_readwrite("nbs", &CanonicalDims::nbs)
        .def_readwrite("nbu", &CanonicalDims::nbu)
        .def_readwrite("nns", &CanonicalDims::nns)
        .def_readwrite("nnu", &CanonicalDims::nnu)
        .def_readwrite("nbe", &CanonicalDims::nbe)
        .def_readwrite("nbi", &CanonicalDims::nbi)
        .def_readwrite("nne", &CanonicalDims::nne)
        .def_readwrite("nni", &CanonicalDims::nni)
        ;

    py::class_<CanonicalMatrixView>(m, "CanonicalMatrixView")
        .def_readonly("dims" , &CanonicalMatrixView::dims)
        .def_readonly("Hss"  , &CanonicalMatrixView::Hss)
        .def_readonly("Hsp"  , &CanonicalMatrixView::Hsp)
        .def_readonly("Vps"  , &CanonicalMatrixView::Vps)
        .def_readonly("Vpp"  , &CanonicalMatrixView::Vpp)
        .def_readonly("Sbsns", &CanonicalMatrixView::Sbsns)
        .def_readonly("Sbsp" , &CanonicalMatrixView::Sbsp)
        .def_readonly("Rbs"  , &CanonicalMatrixView::Rbs)
        .def_readonly("jb"   , &CanonicalMatrixView::jb)
        .def_readonly("jn"   , &CanonicalMatrixView::jn)
        .def_readonly("js"   , &CanonicalMatrixView::js)
        .def_readonly("ju"   , &CanonicalMatrixView::ju)
        ;

    py::class_<CanonicalMatrix>(m, "CanonicalMatrix")
        .def(py::init<Index, Index, Index, Index>())
        .def(py::init<const CanonicalMatrix&>())
        .def("update", &CanonicalMatrix::update)
        .def("view", &CanonicalMatrix::view,
            py::keep_alive<1, 0>(), // keep this object (0) alive while returned object (1) exists
            py::keep_alive<0, 1>()) // keep returned object (1) alive while this object (0) exists
        ;
}
