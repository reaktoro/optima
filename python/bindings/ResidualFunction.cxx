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
#include <Optima/ResidualFunction.hpp>
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
        .def_property_readonly("f", [](const ResidualFunctionResult& self) { return self.f; })
        .def_property_readonly("h", [](const ResidualFunctionResult& self) { return self.h; })
        .def_property_readonly("v", [](const ResidualFunctionResult& self) { return self.v; })
        ;

    py::class_<ResidualFunction>(m, "ResidualFunction")
        .def(py::init<const MasterDims&>())
        .def("initialize"              , &ResidualFunction::initialize)
        .def("update"                  , &ResidualFunction::update)
        .def("updateSkipJacobian"      , &ResidualFunction::updateSkipJacobian)
        .def("canonicalJacobianMatrix" , &ResidualFunction::canonicalJacobianMatrix, py::return_value_policy::reference_internal)
        .def("canonicalResidualVector" , &ResidualFunction::canonicalResidualVector, py::return_value_policy::reference_internal)
        .def("masterJacobianMatrix"    , &ResidualFunction::masterJacobianMatrix, py::return_value_policy::reference_internal)
        .def("masterResidualVector"    , &ResidualFunction::masterResidualVector, py::return_value_policy::reference_internal)
        .def("result"                  , &ResidualFunction::result, py::return_value_policy::reference_internal)
        ;
}
