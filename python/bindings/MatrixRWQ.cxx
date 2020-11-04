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
    py::class_<MatrixViewRWQ>(m, "MatrixViewRWQ")
        .def_readonly("R"  , &MatrixViewRWQ::R)
        .def_readonly("Sbn", &MatrixViewRWQ::Sbn)
        .def_readonly("Sbp", &MatrixViewRWQ::Sbp)
        .def_readonly("jb" , &MatrixViewRWQ::jb)
        .def_readonly("jn" , &MatrixViewRWQ::jn)
        ;

    auto init = [](Index nx, Index np, Index ny, Index nz, MatrixConstRef4py Ax, MatrixConstRef4py Ap)
    {
        return MatrixRWQ(nx, np, ny, nz, Ax, Ap);
    };

    auto update = [](MatrixRWQ& self, MatrixConstRef4py Jx, MatrixConstRef4py Jp, VectorConstRef weights)
    {
        self.update(Jx, Jp, weights);
    };

    py::class_<MatrixRWQ>(m, "MatrixRWQ")
        .def(py::init(init))
        .def(py::init<const MatrixRWQ&>())
        .def("update", update)
        .def("view", &MatrixRWQ::view,
            py::keep_alive<1, 0>(), // keep this object alive while returned object exists
            py::keep_alive<0, 1>()) // keep returned object alive while this object exists
        ;
}
