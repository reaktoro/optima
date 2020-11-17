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
#include <Optima/MasterProblem.hpp>
using namespace Optima;

void exportMasterProblem(py::module& m)
{
    py::class_<MasterProblem>(m, "MasterProblem")
        .def_property_readonly("dims"  , [](const MasterProblem& self) { return self.dims;   })
        .def_property_readonly("Ax"    , [](const MasterProblem& self) { return self.Ax;     })
        .def_property_readonly("Ap"    , [](const MasterProblem& self) { return self.Ap;     })
        .def_property_readonly("f"     , [](const MasterProblem& self) { return self.f;      })
        .def_property_readonly("h"     , [](const MasterProblem& self) { return self.h;      })
        .def_property_readonly("v"     , [](const MasterProblem& self) { return self.v;      })
        .def_property_readonly("b"     , [](const MasterProblem& self) { return self.b;      })
        .def_property_readonly("xlower", [](const MasterProblem& self) { return self.xlower; })
        .def_property_readonly("xupper", [](const MasterProblem& self) { return self.xupper; })
        .def_property_readonly("phi"   , [](const MasterProblem& self) { return self.phi;    })
        ;
}
