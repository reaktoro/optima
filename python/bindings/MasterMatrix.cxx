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
#include <Optima/MasterMatrixOps.hpp>
#include <Optima/MasterVector.hpp>
using namespace Optima;

void exportMasterMatrix(py::module& m)
{
    py::class_<MasterMatrix>(m, "MasterMatrix")
        .def(py::init<MasterDims, MatrixViewH, MatrixViewV, MatrixViewW, MatrixViewRWQ, IndicesView, IndicesView>(),
            keep_argument_alive<0>(),
            keep_argument_alive<1>(),
            keep_argument_alive<2>(),
            keep_argument_alive<3>(),
            keep_argument_alive<4>(),
            keep_argument_alive<5>(),
            keep_argument_alive<6>())
        .def_readonly("dims", &MasterMatrix::dims)
        .def_readonly("H"   , &MasterMatrix::H)
        .def_readonly("V"   , &MasterMatrix::V)
        .def_readonly("W"   , &MasterMatrix::W)
        .def_readonly("RWQ" , &MasterMatrix::RWQ)
        .def_readonly("js"  , &MasterMatrix::js)
        .def_readonly("ju"  , &MasterMatrix::ju)
        .def("__mul__", [](const MasterMatrix& l, const MasterVectorView& r) { return l * r; })
        .def("array", [](const MasterMatrix& self) { return Matrix(self); })
        ;
}
