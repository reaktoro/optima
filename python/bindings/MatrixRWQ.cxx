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
#include <Optima/MatrixRWQ.hpp>
using namespace Optima;

void exportMatrixRWQ(py::module& m)
{
    py::class_<MatrixViewW>(m, "MatrixViewW")
        .def(py::init<MatrixRWQ const&>())
        .def_readonly("Wx" , &MatrixViewW::Wx)
        .def_readonly("Wp" , &MatrixViewW::Wp)
        .def_readonly("Ax" , &MatrixViewW::Ax)
        .def_readonly("Ap" , &MatrixViewW::Ap)
        .def_readonly("Jx" , &MatrixViewW::Jx)
        .def_readonly("Jp" , &MatrixViewW::Jp)
        ;

    py::class_<MatrixViewRWQ>(m, "MatrixViewRWQ")
        .def(py::init<MatrixRWQ const&>())
        .def_readonly("R"  , &MatrixViewRWQ::R)
        .def_readonly("Sbn", &MatrixViewRWQ::Sbn)
        .def_readonly("Sbp", &MatrixViewRWQ::Sbp)
        .def_readonly("jb" , &MatrixViewRWQ::jb)
        .def_readonly("jn" , &MatrixViewRWQ::jn)
        ;

    auto init = [](const MasterDims& dims, MatrixConstRef4py Ax, MatrixConstRef4py Ap)
    {
        return MatrixRWQ(dims, Ax, Ap);
    };

    auto update = [](MatrixRWQ& self, MatrixConstRef4py Jx, MatrixConstRef4py Jp, VectorConstRef weights)
    {
        self.update(Jx, Jp, weights);
    };

    py::class_<MatrixRWQ>(m, "MatrixRWQ")
        .def(py::init(init))
        .def(py::init<const MatrixRWQ&>())
        .def("update", update)
        .def("dims", &MatrixRWQ::dims)
        .def("asMatrixViewW", &MatrixRWQ::asMatrixViewW, py::return_value_policy::reference_internal)
        .def("asMatrixViewRWQ", &MatrixRWQ::asMatrixViewRWQ, py::return_value_policy::reference_internal)
        ;

    py::implicitly_convertible<MatrixRWQ, MatrixViewW>();
    py::implicitly_convertible<MatrixRWQ, MatrixViewRWQ>();
}
