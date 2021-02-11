// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2020 Allan Leal
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
#include <Optima/MasterState.hpp>
using namespace Optima;

void exportMasterState(py::module& m)
{
    auto get_xc = [](const MasterState& s) { return s.xc; };
    auto get_pc = [](const MasterState& s) { return s.pc; };
    auto get_wc = [](const MasterState& s) { return s.wc; };
    auto get_sc = [](const MasterState& s) { return s.sc; };

    auto set_xc = [](MasterState& s, MatrixView4py xc) { s.xc = xc; };
    auto set_pc = [](MasterState& s, MatrixView4py pc) { s.pc = pc; };
    auto set_wc = [](MasterState& s, MatrixView4py wc) { s.wc = wc; };
    auto set_sc = [](MasterState& s, MatrixView4py sc) { s.sc = sc; };

    py::class_<MasterState>(m, "MasterState")
        .def(py::init<>())
        .def_readwrite("u", &MasterState::u)
        .def_readwrite("s", &MasterState::s)
        .def_readwrite("c", &MasterState::c)
        .def_property("xc", get_xc, set_xc)
        .def_property("pc", get_pc, set_pc)
        .def_property("wc", get_wc, set_wc)
        .def_property("sc", get_sc, set_sc)
        ;
}
