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
#include <Optima/ComplementarityVariables.hpp>
using namespace Optima;

void exportComplementarityVariables(py::module& m)
{
    auto wrtCanonicalLowerBounds1 = py::overload_cast<>(&ComplementarityVariables::wrtCanonicalLowerBounds);
    auto wrtCanonicalLowerBounds2 = py::overload_cast<>(&ComplementarityVariables::wrtCanonicalLowerBounds, py::const_);
    auto wrtCanonicalUpperBounds1 = py::overload_cast<>(&ComplementarityVariables::wrtCanonicalUpperBounds);
    auto wrtCanonicalUpperBounds2 = py::overload_cast<>(&ComplementarityVariables::wrtCanonicalUpperBounds, py::const_);
    auto wrtLowerBounds1 = py::overload_cast<>(&ComplementarityVariables::wrtLowerBounds);
    auto wrtLowerBounds2 = py::overload_cast<>(&ComplementarityVariables::wrtLowerBounds, py::const_);
    auto wrtUpperBounds1 = py::overload_cast<>(&ComplementarityVariables::wrtUpperBounds);
    auto wrtUpperBounds2 = py::overload_cast<>(&ComplementarityVariables::wrtUpperBounds, py::const_);
    auto wrtLinearInequalityConstraints1 = py::overload_cast<>(&ComplementarityVariables::wrtLinearInequalityConstraints);
    auto wrtLinearInequalityConstraints2 = py::overload_cast<>(&ComplementarityVariables::wrtLinearInequalityConstraints, py::const_);
    auto wrtNonLinearInequalityConstraints1 = py::overload_cast<>(&ComplementarityVariables::wrtNonLinearInequalityConstraints);
    auto wrtNonLinearInequalityConstraints2 = py::overload_cast<>(&ComplementarityVariables::wrtNonLinearInequalityConstraints, py::const_);

    auto rvp = py::return_value_policy::reference_internal;

    py::class_<ComplementarityVariables>(m, "ComplementarityVariables")
        .def(py::init<>())
        .def(py::init<const Constraints&>())
        .def("wrtCanonicalLowerBounds", wrtCanonicalLowerBounds1, rvp, "Return the complementarity variables with respect to the lower bound constraints of the canonical optimization problem.")
        .def("wrtCanonicalLowerBounds", wrtCanonicalLowerBounds2, rvp, "Return the complementarity variables with respect to the lower bound constraints of the canonical optimization problem.")
        .def("wrtCanonicalUpperBounds", wrtCanonicalUpperBounds1, rvp, "Return the complementarity variables with respect to the upper bound constraints of the canonical optimization problem.")
        .def("wrtCanonicalUpperBounds", wrtCanonicalUpperBounds2, rvp, "Return the complementarity variables with respect to the upper bound constraints of the canonical optimization problem.")
        .def("wrtLowerBounds", wrtLowerBounds1, rvp, "Return the complementarity variables with respect to the lower bound constraints of the original optimization problem.")
        .def("wrtLowerBounds", wrtLowerBounds2, rvp, "Return the complementarity variables with respect to the lower bound constraints of the original optimization problem.")
        .def("wrtUpperBounds", wrtUpperBounds1, rvp, "Return the complementarity variables with respect to the upper bound constraints of the original optimization problem.")
        .def("wrtUpperBounds", wrtUpperBounds2, rvp, "Return the complementarity variables with respect to the upper bound constraints of the original optimization problem.")
        .def("wrtLinearInequalityConstraints", wrtLinearInequalityConstraints1, rvp, "Return the complementarity variables with respect to the linear inequality constraints of the original optimization problem.")
        .def("wrtLinearInequalityConstraints", wrtLinearInequalityConstraints2, rvp, "Return the complementarity variables with respect to the linear inequality constraints of the original optimization problem.")
        .def("wrtNonLinearInequalityConstraints", wrtNonLinearInequalityConstraints1, rvp, "Return the complementarity variables with respect to the non-linear inequality constraints of the original optimization problem.")
        .def("wrtNonLinearInequalityConstraints", wrtNonLinearInequalityConstraints2, rvp, "Return the complementarity variables with respect to the non-linear inequality constraints of the original optimization problem.")
        ;
}
