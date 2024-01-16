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
#include <Optima/ResidualFunction.hpp>
#include <Optima/Stability.hpp>
using namespace Optima;

void exportResidualFunction(py::module& m)
{
    py::class_<ResidualFunctionUpdateStatus>(m, "ResidualFunctionUpdateStatus")
        .def(py::init<>())
        .def_readwrite("f", &ResidualFunctionUpdateStatus::f)
        .def_readwrite("h", &ResidualFunctionUpdateStatus::h)
        .def_readwrite("v", &ResidualFunctionUpdateStatus::v)
        ;

    py::class_<ResidualFunctionResult>(m, "ResidualFunctionResult")
        .def_property_readonly("f",               [](const ResidualFunctionResult& self) { return self.f;               })
        .def_property_readonly("h",               [](const ResidualFunctionResult& self) { return self.h;               })
        .def_property_readonly("v",               [](const ResidualFunctionResult& self) { return self.v;               })
        .def_property_readonly("Jm",              [](const ResidualFunctionResult& self) { return self.Jm;              })
        .def_property_readonly("Jc",              [](const ResidualFunctionResult& self) { return self.Jc;              })
        .def_property_readonly("Fm",              [](const ResidualFunctionResult& self) { return self.Fm;              })
        .def_property_readonly("Fc",              [](const ResidualFunctionResult& self) { return self.Fc;              })
        .def_property_readonly("stabilitystatus", [](const ResidualFunctionResult& self) { return self.stabilitystatus; })
        .def_property_readonly("succeeded",       [](const ResidualFunctionResult& self) { return self.succeeded;       })
        ;

    py::class_<ResidualFunction>(m, "ResidualFunction")
        .def(py::init<>())
        .def("initialize"                  , &ResidualFunction::initialize)
        .def("update"                      , &ResidualFunction::update)
        .def("updateSkipJacobian"          , &ResidualFunction::updateSkipJacobian)
        .def("updateOnlyJacobian"          , &ResidualFunction::updateOnlyJacobian)
        .def("result"                      , &ResidualFunction::result, py::return_value_policy::reference_internal)
        ;
}
