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
#include <Optima/MasterMatrixOps.hpp>
#include <Optima/MasterVector.hpp>
using namespace Optima;

/// Used to indicate that the k-th argument in a method should be kept alive in Python.
/// Note: pybind11's numbering convention for `py::keep_alive` starts with `2`
/// for arguments. Index `1` denotes the `this` pointer and `0` the returned object.
/// Here, however, `0` denotes the first argument in C++, which is not the `this` pointer.
template<size_t k>
using keep_argument_alive = py::keep_alive<1, k + 2>;

void exportMasterMatrix(py::module& m)
{
    py::class_<MatrixViewH>(m, "MatrixViewH")
        .def(py::init<MatrixConstRef4py, MatrixConstRef4py, bool>(),
            keep_argument_alive<0>(),
            keep_argument_alive<1>())
        .def_readonly("Hxx"      , &MatrixViewH::Hxx)
        .def_readonly("Hxp"      , &MatrixViewH::Hxp)
        .def_readonly("isHxxDiag", &MatrixViewH::isHxxDiag)
        ;

    py::class_<MatrixViewV>(m, "MatrixViewV")
        .def(py::init<MatrixConstRef4py, MatrixConstRef4py>(),
            keep_argument_alive<0>(),
            keep_argument_alive<1>())
        .def_readonly("Vpx", &MatrixViewV::Vpx)
        .def_readonly("Vpp", &MatrixViewV::Vpp)
        ;

    py::class_<MatrixViewW>(m, "MatrixViewW")
        .def(py::init<MatrixConstRef4py, MatrixConstRef4py>(),
            keep_argument_alive<0>(),
            keep_argument_alive<1>())
        .def_readonly("Wx", &MatrixViewW::Wx)
        .def_readonly("Wp", &MatrixViewW::Wp)
        ;

    py::class_<MasterMatrix>(m, "MasterMatrix")
        .def(py::init<MatrixViewH, MatrixViewV, MatrixViewW, MatrixViewRWQ, IndicesConstRef, IndicesConstRef>(),
            keep_argument_alive<0>(),
            keep_argument_alive<1>(),
            keep_argument_alive<2>(),
            keep_argument_alive<3>(),
            keep_argument_alive<4>(),
            keep_argument_alive<5>())
        .def_readonly("H"  , &MasterMatrix::H)
        .def_readonly("V"  , &MasterMatrix::V)
        .def_readonly("W"  , &MasterMatrix::W)
        .def_readonly("RWQ", &MasterMatrix::RWQ)
        .def_readonly("js" , &MasterMatrix::js)
        .def_readonly("ju" , &MasterMatrix::ju)
        .def("__mul__", [](const MasterMatrix& l, const MasterVectorView& r) { return l * r; })
        .def("array", [](const MasterMatrix& self) { return Matrix(self); })
        ;
}
