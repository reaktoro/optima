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
#include <Optima/JacobianBlockH.hpp>
#include <Optima/JacobianBlockV.hpp>
#include <Optima/JacobianBlockW.hpp>
#include <Optima/JacobianMatrix.hpp>
using namespace Optima;

void exportJacobianMatrix(py::module& m)
{
    py::class_<JacobianMatrix::Dims>(m, "JacobianMatrixDims")
        .def(py::init<>())
        .def_readwrite("nx", &JacobianMatrix::Dims::nx)
        .def_readwrite("np", &JacobianMatrix::Dims::np)
        .def_readwrite("ny", &JacobianMatrix::Dims::ny)
        .def_readwrite("nz", &JacobianMatrix::Dims::nz)
        .def_readwrite("nw", &JacobianMatrix::Dims::nw)
        .def_readwrite("ns", &JacobianMatrix::Dims::ns)
        .def_readwrite("nu", &JacobianMatrix::Dims::nu)
        .def_readwrite("nb", &JacobianMatrix::Dims::nb)
        .def_readwrite("nn", &JacobianMatrix::Dims::nn)
        .def_readwrite("nl", &JacobianMatrix::Dims::nl)
        .def_readwrite("nbs", &JacobianMatrix::Dims::nbs)
        .def_readwrite("nbu", &JacobianMatrix::Dims::nbu)
        .def_readwrite("nns", &JacobianMatrix::Dims::nns)
        .def_readwrite("nnu", &JacobianMatrix::Dims::nnu)
        .def_readwrite("nbe", &JacobianMatrix::Dims::nbe)
        .def_readwrite("nbi", &JacobianMatrix::Dims::nbi)
        .def_readwrite("nne", &JacobianMatrix::Dims::nne)
        .def_readwrite("nni", &JacobianMatrix::Dims::nni)
        ;

    py::class_<JacobianMatrix::CanonicalForm>(m, "JacobianMatrixCanonicalForm")
        .def_readonly("Hss", &JacobianMatrix::CanonicalForm::Hss)
        .def_readonly("Hsp", &JacobianMatrix::CanonicalForm::Hsp)
        .def_readonly("Vps", &JacobianMatrix::CanonicalForm::Vps)
        .def_readonly("Vpp", &JacobianMatrix::CanonicalForm::Vpp)
        .def_readonly("Sbn", &JacobianMatrix::CanonicalForm::Sbn)
        .def_readonly("Sbp", &JacobianMatrix::CanonicalForm::Sbp)
        .def_readonly("R", &JacobianMatrix::CanonicalForm::R)
        .def_readonly("Ws", &JacobianMatrix::CanonicalForm::Ws)
        .def_readonly("Wu", &JacobianMatrix::CanonicalForm::Wu)
        .def_readonly("Wp", &JacobianMatrix::CanonicalForm::Wp)
        .def_readonly("As", &JacobianMatrix::CanonicalForm::As)
        .def_readonly("Au", &JacobianMatrix::CanonicalForm::Au)
        .def_readonly("Ap", &JacobianMatrix::CanonicalForm::Ap)
        .def_readonly("Js", &JacobianMatrix::CanonicalForm::Js)
        .def_readonly("Ju", &JacobianMatrix::CanonicalForm::Ju)
        .def_readonly("Jp", &JacobianMatrix::CanonicalForm::Jp)
        .def_readonly("jb", &JacobianMatrix::CanonicalForm::jb)
        .def_readonly("jn", &JacobianMatrix::CanonicalForm::jn)
        .def_readonly("js", &JacobianMatrix::CanonicalForm::js)
        .def_readonly("ju", &JacobianMatrix::CanonicalForm::ju)
        ;

    py::class_<JacobianMatrix>(m, "JacobianMatrix")
        .def(py::init<Index, Index, Index, Index>())
        .def(py::init<const JacobianMatrix&>())
        .def("update", &JacobianMatrix::update)
        .def("dims", &JacobianMatrix::dims)
        .def("canonicalForm", &JacobianMatrix::canonicalForm)
        ;
}
