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
#include <Optima/Problem.hpp>
using namespace Optima;

void exportProblem(py::module& m)
{
    auto set_r     = [](Problem& self, ResourcesFunction r) { self.r = r; };
    auto set_f     = [](Problem& self, ObjectiveFunction f) { self.f = f; };
    auto set_he    = [](Problem& self, ConstraintFunction he) { self.he = he; };
    auto set_hg    = [](Problem& self, ConstraintFunction hg) { self.hg = hg; };
    auto set_v     = [](Problem& self, ConstraintFunction v) { self.v = v; };
    auto set_Aex   = [](Problem& self, MatrixView4py Aex) { self.Aex = Aex; };
    auto set_Aep   = [](Problem& self, MatrixView4py Aep) { self.Aep = Aep; };
    auto set_Agx   = [](Problem& self, MatrixView4py Agx) { self.Agx = Agx; };
    auto set_Agp   = [](Problem& self, MatrixView4py Agp) { self.Agp = Agp; };
    auto set_bec   = [](Problem& self, MatrixView4py bec) { self.bec = bec; };
    auto set_bgc   = [](Problem& self, MatrixView4py bgc) { self.bgc = bgc; };

    auto get_r     = [](Problem& self) { return self.r; };
    auto get_f     = [](Problem& self) { return self.f; };
    auto get_he    = [](Problem& self) { return self.he; };
    auto get_hg    = [](Problem& self) { return self.hg; };
    auto get_v     = [](Problem& self) { return self.v; };
    auto get_Aex   = [](Problem& self) { return self.Aex; };
    auto get_Aep   = [](Problem& self) { return self.Aep; };
    auto get_Agx   = [](Problem& self) { return self.Agx; };
    auto get_Agp   = [](Problem& self) { return self.Agp; };
    auto get_bec   = [](Problem& self) { return self.bec; };
    auto get_bgc   = [](Problem& self) { return self.bgc; };

    py::class_<Problem>(m, "Problem")
        .def(py::init<const Dims&>())
        .def_readonly("dims"    , &Problem::dims)
        .def_property("r"       , get_r, set_r)
        .def_property("f"       , get_f, set_f)
        .def_property("he"      , get_he, set_he)
        .def_property("hg"      , get_hg, set_hg)
        .def_property("v"       , get_v, set_v)
        .def_property("Aex"     , get_Aex, set_Aex)
        .def_property("Aep"     , get_Aep, set_Aep)
        .def_property("Agx"     , get_Agx, set_Agx)
        .def_property("Agp"     , get_Agp, set_Agp)
        .def_readwrite("be"     , &Problem::be)
        .def_readwrite("bg"     , &Problem::bg)
        .def_readwrite("xlower" , &Problem::xlower)
        .def_readwrite("xupper" , &Problem::xupper)
        .def_readwrite("plower" , &Problem::plower)
        .def_readwrite("pupper" , &Problem::pupper)
        .def_readwrite("c"      , &Problem::c)
        .def_property("bec"     , get_bec, set_bec)
        .def_property("bgc"     , get_bgc, set_bgc)
        ;
}
