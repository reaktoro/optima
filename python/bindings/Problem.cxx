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
#include <pybind11/functional.h>
namespace py = pybind11;

// Optima includes
#include <Optima/Problem.hpp>
using namespace Optima;

void exportProblem(py::module& m)
{
    auto set_Aex = [](Problem& self, MatrixView4py  Aex) { self.Aex = Aex; };
    auto set_Aep = [](Problem& self, MatrixView4py  Aep) { self.Aep = Aep; };
    auto set_Agx = [](Problem& self, MatrixView4py  Agx) { self.Agx = Agx; };
    auto set_Agp = [](Problem& self, MatrixView4py  Agp) { self.Agp = Agp; };
    auto set_he  = [](Problem& self, ConstraintFunction  he) { self.he  = he;  };
    auto set_hg  = [](Problem& self, ConstraintFunction  hg) { self.hg  = hg;  };
    auto set_v   = [](Problem& self, ConstraintFunction   v) { self.v   = v;   };
    auto set_f   = [](Problem& self, ObjectiveFunction    f) { self.f   = f;   };
    auto set_fxw = [](Problem& self, MatrixView4py  fxw) { self.fxw = fxw; };
    auto set_bw  = [](Problem& self, MatrixView4py   bw) { self.bw  = bw;  };
    auto set_hw  = [](Problem& self, MatrixView4py   hw) { self.hw  = hw;  };
    auto set_vw  = [](Problem& self, MatrixView4py   vw) { self.vw  = vw;  };

    auto get_Aex = [](Problem& self) { return self.Aex; };
    auto get_Aep = [](Problem& self) { return self.Aep; };
    auto get_Agx = [](Problem& self) { return self.Agx; };
    auto get_Agp = [](Problem& self) { return self.Agp; };
    auto get_he  = [](Problem& self) { return self.he;  };
    auto get_hg  = [](Problem& self) { return self.hg;  };
    auto get_v   = [](Problem& self) { return self.v;   };
    auto get_f   = [](Problem& self) { return self.f;   };
    auto get_fxw = [](Problem& self) { return self.fxw; };
    auto get_bw  = [](Problem& self) { return self.bw;  };
    auto get_hw  = [](Problem& self) { return self.hw;  };
    auto get_vw  = [](Problem& self) { return self.vw;  };

    py::class_<Problem>(m, "Problem")
        .def(py::init<const Dims&>())
        .def_readonly("dims", &Problem::dims)
        .def_property("Aex", get_Aex, set_Aex)
        .def_property("Aep", get_Aep, set_Aep)
        .def_property("Agx", get_Agx, set_Agx)
        .def_property("Agp", get_Agp, set_Agp)
        .def_readwrite("be", &Problem::be)
        .def_readwrite("bg", &Problem::bg)
        .def_property("he", get_he, set_he)
        .def_property("hg", get_hg, set_hg)
        .def_property("v", get_v, set_v)
        .def_property("f", get_f, set_f)
        .def_readwrite("xlower", &Problem::xlower)
        .def_readwrite("xupper", &Problem::xupper)
        .def_readwrite("plower", &Problem::plower)
        .def_readwrite("pupper", &Problem::pupper)
        .def_property("fxw", get_fxw, set_fxw)
        .def_property("bw", get_bw, set_bw)
        .def_property("hw", get_hw, set_hw)
        .def_property("vw", get_vw, set_vw)
        ;
}
