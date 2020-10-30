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
#include <Optima/MasterMatrixH.hpp>
using namespace Optima;

void exportMasterMatrixH(py::module& m)
{
    auto init = [](MatrixConstRef4py Hxx, MatrixConstRef4py Hxp)
    {
        return MasterMatrixH(Hxx, Hxp);
    };

    auto set_Hxx = [](MasterMatrixH& self, MatrixConstRef4py Hxx) { self.Hxx = Hxx; };
    auto set_Hxp = [](MasterMatrixH& self, MatrixConstRef4py Hxp) { self.Hxp = Hxp; };

    auto get_Hxx = [](const MasterMatrixH& self) { return self.Hxx; };
    auto get_Hxp = [](const MasterMatrixH& self) { return self.Hxp; };

    py::class_<MasterMatrixH>(m, "MasterMatrixH")
        .def(py::init<Index, Index>())
        .def(py::init(init))
        .def(py::init<const MasterMatrixH&>())
        .def("isHxxDiagonal", py::overload_cast<>(&MasterMatrixH::isHxxDiagonal, py::const_))
        .def("isHxxDiagonal", py::overload_cast<bool>(&MasterMatrixH::isHxxDiagonal))
        .def_property("Hxx", get_Hxx, set_Hxx)
        .def_property("Hxp", get_Hxp, set_Hxp)
        ;
}
