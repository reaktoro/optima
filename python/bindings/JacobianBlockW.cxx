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
#include <Optima/JacobianBlockW.hpp>
using namespace Optima;

void exportJacobianBlockW(py::module& m)
{
    py::class_<JacobianBlockW::CanonicalForm>(m, "JacobianBlockWCanonicalForm")
        .def_readonly("R", &JacobianBlockW::CanonicalForm::R)
        .def_readonly("Sbn", &JacobianBlockW::CanonicalForm::Sbn)
        .def_readonly("Sbp", &JacobianBlockW::CanonicalForm::Sbp)
        .def_readonly("jb", &JacobianBlockW::CanonicalForm::jb)
        .def_readonly("jn", &JacobianBlockW::CanonicalForm::jn)
        ;

    auto init = [](Index nx, Index np, Index ny, Index nz, MatrixConstRef4py Ax, MatrixConstRef4py Ap)
    {
        return JacobianBlockW(nx, np, ny, nz, Ax, Ap);
    };

    auto update = [](JacobianBlockW& self, MatrixConstRef4py Jx, MatrixConstRef4py Jp, VectorConstRef weights)
    {
        self.update(Jx, Jp, weights);
    };

    py::class_<JacobianBlockW>(m, "JacobianBlockW")
        .def(py::init(init))
        .def(py::init<const JacobianBlockW&>())
        .def("update", update)
        .def("canonicalForm", &JacobianBlockW::canonicalForm)
        .def_readonly("Ax", &JacobianBlockW::Ax)
        .def_readonly("Ap", &JacobianBlockW::Ap)
        .def_readonly("Jx", &JacobianBlockW::Jx)
        .def_readonly("Jp", &JacobianBlockW::Jp)
        .def_readonly("Wx", &JacobianBlockW::Wx)
        .def_readonly("Wp", &JacobianBlockW::Wp)
        ;
}
