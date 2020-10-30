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
#include <pybind11/operators.h>
namespace py = pybind11;

// Optima includes
#include <Optima/MasterMatrix.hpp>
#include <Optima/MasterMatrixH.hpp>
#include <Optima/MasterMatrixV.hpp>
#include <Optima/MasterMatrixW.hpp>
#include <Optima/MasterMatrixOperators.hpp>
#include <Optima/MasterVector.hpp>
using namespace Optima;

void exportMasterMatrix(py::module& m)
{
    py::class_<CanonicalDims>(m, "CanonicalDims")
        .def(py::init<>())
        .def_readwrite("nx", &CanonicalDims::nx)
        .def_readwrite("np", &CanonicalDims::np)
        .def_readwrite("ny", &CanonicalDims::ny)
        .def_readwrite("nz", &CanonicalDims::nz)
        .def_readwrite("nw", &CanonicalDims::nw)
        .def_readwrite("ns", &CanonicalDims::ns)
        .def_readwrite("nu", &CanonicalDims::nu)
        .def_readwrite("nb", &CanonicalDims::nb)
        .def_readwrite("nn", &CanonicalDims::nn)
        .def_readwrite("nl", &CanonicalDims::nl)
        .def_readwrite("nbs", &CanonicalDims::nbs)
        .def_readwrite("nbu", &CanonicalDims::nbu)
        .def_readwrite("nns", &CanonicalDims::nns)
        .def_readwrite("nnu", &CanonicalDims::nnu)
        .def_readwrite("nbe", &CanonicalDims::nbe)
        .def_readwrite("nbi", &CanonicalDims::nbi)
        .def_readwrite("nne", &CanonicalDims::nne)
        .def_readwrite("nni", &CanonicalDims::nni)
        ;

    py::class_<CanonicalMatrix>(m, "CanonicalMatrix")
        .def_readonly("dims", &CanonicalMatrix::dims)
        .def_readonly("Hss", &CanonicalMatrix::Hss)
        .def_readonly("Hsp", &CanonicalMatrix::Hsp)
        .def_readonly("Vps", &CanonicalMatrix::Vps)
        .def_readonly("Vpp", &CanonicalMatrix::Vpp)
        .def_readonly("Sbn", &CanonicalMatrix::Sbsns)
        .def_readonly("Sbp", &CanonicalMatrix::Sbsp)
        ;

    py::class_<CanonicalDetails>(m, "CanonicalDetails")
        .def_readonly("dims", &CanonicalDetails::dims)
        .def_readonly("Hss", &CanonicalDetails::Hss)
        .def_readonly("Hsp", &CanonicalDetails::Hsp)
        .def_readonly("Vps", &CanonicalDetails::Vps)
        .def_readonly("Vpp", &CanonicalDetails::Vpp)
        .def_readonly("Sbn", &CanonicalDetails::Sbn)
        .def_readonly("Sbp", &CanonicalDetails::Sbp)
        .def_readonly("R", &CanonicalDetails::R)
        .def_readonly("Ws", &CanonicalDetails::Ws)
        .def_readonly("Wu", &CanonicalDetails::Wu)
        .def_readonly("Wp", &CanonicalDetails::Wp)
        .def_readonly("As", &CanonicalDetails::As)
        .def_readonly("Au", &CanonicalDetails::Au)
        .def_readonly("Ap", &CanonicalDetails::Ap)
        .def_readonly("Js", &CanonicalDetails::Js)
        .def_readonly("Ju", &CanonicalDetails::Ju)
        .def_readonly("Jp", &CanonicalDetails::Jp)
        .def_readonly("jb", &CanonicalDetails::jb)
        .def_readonly("jn", &CanonicalDetails::jn)
        .def_readonly("js", &CanonicalDetails::js)
        .def_readonly("ju", &CanonicalDetails::ju)
        ;

    py::class_<MasterMatrix>(m, "MasterMatrix")
        .def(py::init<Index, Index, Index, Index>())
        .def(py::init<const MasterMatrix&>())
        .def("update", &MasterMatrix::update)
        .def("canonicalMatrix", &MasterMatrix::canonicalMatrix)
        .def("canonicalForm", &MasterMatrix::canonicalForm)
        .def("matrix", &MasterMatrix::matrix)
        .def("__mul__", [](const MasterMatrix& l, const MasterVector& r) { return l * r; })
        ;
}
