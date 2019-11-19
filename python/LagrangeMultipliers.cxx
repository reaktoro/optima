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
#include <Optima/Constraints.hpp>
#include <Optima/LagrangeMultipliers.hpp>
using namespace Optima;

void exportLagrangeMultipliers(py::module& m)
{
    auto canonical1 = py::overload_cast<>(&LagrangeMultipliers::canonical);
    auto canonical2 = py::overload_cast<>(&LagrangeMultipliers::canonical, py::const_);
    auto wrtLinearEqualityConstraints1 = py::overload_cast<>(&LagrangeMultipliers::wrtLinearEqualityConstraints);
    auto wrtLinearEqualityConstraints2 = py::overload_cast<>(&LagrangeMultipliers::wrtLinearEqualityConstraints, py::const_);
    auto wrtNonLinearEqualityConstraints1 = py::overload_cast<>(&LagrangeMultipliers::wrtNonLinearEqualityConstraints);
    auto wrtNonLinearEqualityConstraints2 = py::overload_cast<>(&LagrangeMultipliers::wrtNonLinearEqualityConstraints, py::const_);
    auto wrtLinearInequalityConstraints1 = py::overload_cast<>(&LagrangeMultipliers::wrtLinearInequalityConstraints);
    auto wrtLinearInequalityConstraints2 = py::overload_cast<>(&LagrangeMultipliers::wrtLinearInequalityConstraints, py::const_);
    auto wrtNonLinearInequalityConstraints1 = py::overload_cast<>(&LagrangeMultipliers::wrtNonLinearInequalityConstraints);
    auto wrtNonLinearInequalityConstraints2 = py::overload_cast<>(&LagrangeMultipliers::wrtNonLinearInequalityConstraints, py::const_);

    auto rvp = py::return_value_policy::reference_internal;

    py::class_<LagrangeMultipliers>(m, "LagrangeMultipliers")
        .def(py::init<>())
        .def(py::init<const Constraints&>())
        .def("canonical", canonical1, rvp, "Return the Lagrange multipliers of the canonical optimization problem.")
        .def("canonical", canonical2, rvp, "Return the Lagrange multipliers of the canonical optimization problem.")
        .def("wrtLinearEqualityConstraints", wrtLinearEqualityConstraints1, rvp, "Return the Lagrange multipliers with respect to the linear equality constraints of the original optimization problem.")
        .def("wrtLinearEqualityConstraints", wrtLinearEqualityConstraints2, rvp, "Return the Lagrange multipliers with respect to the linear equality constraints of the original optimization problem.")
        .def("wrtNonLinearEqualityConstraints", wrtNonLinearEqualityConstraints1, rvp, "Return the Lagrange multipliers with respect to the non-linear equality constraints of the original optimization problem.")
        .def("wrtNonLinearEqualityConstraints", wrtNonLinearEqualityConstraints2, rvp, "Return the Lagrange multipliers with respect to the non-linear equality constraints of the original optimization problem.")
        .def("wrtLinearInequalityConstraints", wrtLinearInequalityConstraints1, rvp, "Return the Lagrange multipliers with respect to the linear inequality constraints of the original optimization problem.")
        .def("wrtLinearInequalityConstraints", wrtLinearInequalityConstraints2, rvp, "Return the Lagrange multipliers with respect to the linear inequality constraints of the original optimization problem.")
        .def("wrtNonLinearInequalityConstraints", wrtNonLinearInequalityConstraints1, rvp, "Return the Lagrange multipliers with respect to the non-linear inequality constraints of the original optimization problem.")
        .def("wrtNonLinearInequalityConstraints", wrtNonLinearInequalityConstraints2, rvp, "Return the Lagrange multipliers with respect to the non-linear inequality constraints of the original optimization problem.")
        ;
}
