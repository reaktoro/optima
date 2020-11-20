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
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

// Optima includes
#include <Optima/MasterProblem.hpp>
using namespace Optima;

void exportMasterProblem(py::module& m)
{
    auto get_Ax = [](const MasterProblem& s) { return s.Ax; }; // TODO: Create macros in pybindx.hpp to create these get/set methods.
    auto get_Ap = [](const MasterProblem& s) { return s.Ap; };
    auto set_Ax = [](MasterProblem& s, MatrixView4py Ax) { s.Ax = Ax; };
    auto set_Ap = [](MasterProblem& s, MatrixView4py Ap) { s.Ap = Ap; };

    py::class_<MasterProblem>(m, "MasterProblem")
        .def(py::init<>())
        .def_readwrite("f"     , &MasterProblem::f)
        .def_readwrite("h"     , &MasterProblem::h)
        .def_readwrite("v"     , &MasterProblem::v)
        .def_property("Ax"     , get_Ax, set_Ax)
        .def_property("Ap"     , get_Ap, set_Ap)
        .def_readwrite("b"     , &MasterProblem::b)
        .def_readwrite("xlower", &MasterProblem::xlower)
        .def_readwrite("xupper", &MasterProblem::xupper)
        .def_readwrite("phi"   , &MasterProblem::phi)
        ;
}
