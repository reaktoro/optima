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
#include <Optima/MasterMatrixV.hpp>
using namespace Optima;

void exportMasterMatrixV(py::module& m)
{
    auto init = [](MatrixConstRef4py Vpx, MatrixConstRef4py Vpp)
    {
        return MasterMatrixV(Vpx, Vpp);
    };

    auto set_Vpx = [](MasterMatrixV& self, MatrixConstRef4py Vpx) { self.Vpx = Vpx; };
    auto set_Vpp = [](MasterMatrixV& self, MatrixConstRef4py Vpp) { self.Vpp = Vpp; };

    auto get_Vpx = [](const MasterMatrixV& self) { return self.Vpx; };
    auto get_Vpp = [](const MasterMatrixV& self) { return self.Vpp; };

    py::class_<MasterMatrixV>(m, "MasterMatrixV")
        .def(py::init<Index, Index>())
        .def(py::init(init))
        .def(py::init<const MasterMatrixV&>())
        .def_property("Vpx", get_Vpx, set_Vpx)
        .def_property("Vpp", get_Vpp, set_Vpp)
        ;
}
