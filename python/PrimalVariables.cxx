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
#include <Optima/PrimalVariables.hpp>
using namespace Optima;

void exportPrimalVariables(py::module& m)
{
    auto canonicalPrimalVariables1 = py::overload_cast<>(&PrimalVariables::canonicalPrimalVariables);
    auto canonicalPrimalVariables2 = py::overload_cast<>(&PrimalVariables::canonicalPrimalVariables, py::const_);
    auto originalPrimalVariables1 = py::overload_cast<>(&PrimalVariables::originalPrimalVariables);
    auto originalPrimalVariables2 = py::overload_cast<>(&PrimalVariables::originalPrimalVariables, py::const_);
    auto slackVariablesLinearInequalityConstraints1 = py::overload_cast<>(&PrimalVariables::slackVariablesLinearInequalityConstraints);
    auto slackVariablesLinearInequalityConstraints2 = py::overload_cast<>(&PrimalVariables::slackVariablesLinearInequalityConstraints, py::const_);
    auto slackVariablesNonLinearInequalityConstraints1 = py::overload_cast<>(&PrimalVariables::slackVariablesNonLinearInequalityConstraints);
    auto slackVariablesNonLinearInequalityConstraints2 = py::overload_cast<>(&PrimalVariables::slackVariablesNonLinearInequalityConstraints, py::const_);

    auto rvp = py::return_value_policy::reference_internal;

    py::class_<PrimalVariables>(m, "PrimalVariables")
        .def(py::init<>())
        .def(py::init<const Constraints&>())
        .def("canonicalPrimalVariables", canonicalPrimalVariables1, rvp, "Return the primal variables of the canonical optimization problem.")
        .def("canonicalPrimalVariables", canonicalPrimalVariables2, rvp, "Return the primal variables of the canonical optimization problem.")
        .def("originalPrimalVariables", originalPrimalVariables1, rvp, "Return the primal variables of the original optimization problem.")
        .def("originalPrimalVariables", originalPrimalVariables2, rvp, "Return the primal variables of the original optimization problem.")
        .def("slackVariablesLinearInequalityConstraints", slackVariablesLinearInequalityConstraints1, rvp, "Return the primal variables of the canonical optimization problem with respect to linear inequality constraints.")
        .def("slackVariablesLinearInequalityConstraints", slackVariablesLinearInequalityConstraints2, rvp, "Return the primal variables of the canonical optimization problem with respect to linear inequality constraints.")
        .def("slackVariablesNonLinearInequalityConstraints", slackVariablesNonLinearInequalityConstraints1, rvp, "Return the primal variables of the canonical optimization problem with respect to non-linear inequality constraints.")
        .def("slackVariablesNonLinearInequalityConstraints", slackVariablesNonLinearInequalityConstraints2, rvp, "Return the primal variables of the canonical optimization problem with respect to non-linear inequality constraints.")
        ;
}
