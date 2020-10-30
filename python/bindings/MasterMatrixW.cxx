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
#include <Optima/MasterMatrixW.hpp>
using namespace Optima;

void exportMasterMatrixW(py::module& m)
{
    py::class_<MasterMatrixW::EchelonForm>(m, "MasterMatrixWEchelonForm")
        .def_readonly("R", &MasterMatrixW::EchelonForm::R)
        .def_readonly("Sbn", &MasterMatrixW::EchelonForm::Sbn)
        .def_readonly("Sbp", &MasterMatrixW::EchelonForm::Sbp)
        .def_readonly("jb", &MasterMatrixW::EchelonForm::jb)
        .def_readonly("jn", &MasterMatrixW::EchelonForm::jn)
        ;

    auto init = [](Index nx, Index np, Index ny, Index nz, MatrixConstRef4py Ax, MatrixConstRef4py Ap)
    {
        return MasterMatrixW(nx, np, ny, nz, Ax, Ap);
    };

    auto update = [](MasterMatrixW& self, MatrixConstRef4py Jx, MatrixConstRef4py Jp, VectorConstRef weights)
    {
        self.update(Jx, Jp, weights);
    };

    py::class_<MasterMatrixW>(m, "MasterMatrixW")
        .def(py::init(init))
        .def(py::init<const MasterMatrixW&>())
        .def("update", update)
        .def("echelonForm", &MasterMatrixW::echelonForm)
        .def_readonly("Ax", &MasterMatrixW::Ax)
        .def_readonly("Ap", &MasterMatrixW::Ap)
        .def_readonly("Jx", &MasterMatrixW::Jx)
        .def_readonly("Jp", &MasterMatrixW::Jp)
        .def_readonly("Wx", &MasterMatrixW::Wx)
        .def_readonly("Wp", &MasterMatrixW::Wp)
        ;
}
