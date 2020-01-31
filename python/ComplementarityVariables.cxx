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

// // Optima includes
// #include <Optima/Constraints.hpp>
// #include <Optima/ComplementarityVariables.hpp>
// using namespace Optima;

void exportComplementarityVariables(py::module& m)
{
    // auto canonical1 = py::overload_cast<>(&ComplementarityVariables::canonical);
    // auto canonical2 = py::overload_cast<>(&ComplementarityVariables::canonical, py::const_);
    // auto original1 = py::overload_cast<>(&ComplementarityVariables::original);
    // auto original2 = py::overload_cast<>(&ComplementarityVariables::original, py::const_);
    // auto wrtLinearInequalityConstraints1 = py::overload_cast<>(&ComplementarityVariables::wrtLinearInequalityConstraints);
    // auto wrtLinearInequalityConstraints2 = py::overload_cast<>(&ComplementarityVariables::wrtLinearInequalityConstraints, py::const_);
    // auto wrtNonLinearInequalityConstraints1 = py::overload_cast<>(&ComplementarityVariables::wrtNonLinearInequalityConstraints);
    // auto wrtNonLinearInequalityConstraints2 = py::overload_cast<>(&ComplementarityVariables::wrtNonLinearInequalityConstraints, py::const_);

    // auto rvp = py::return_value_policy::reference_internal;

    // py::class_<ComplementarityVariables>(m, "ComplementarityVariables")
    //     .def(py::init<>())
    //     .def(py::init<const Constraints&>())
    //     .def("canonical", canonical1, rvp, "Return the complementarity variables with respect to the bound constraints of the canonical optimization problem.")
    //     .def("canonical", canonical2, rvp, "Return the complementarity variables with respect to the bound constraints of the canonical optimization problem.")
    //     .def("original", original1, rvp, "Return the complementarity variables with respect to the bound constraints of the original optimization problem.")
    //     .def("original", original2, rvp, "Return the complementarity variables with respect to the bound constraints of the original optimization problem.")
    //     .def("wrtLinearInequalityConstraints", wrtLinearInequalityConstraints1, rvp, "Return the complementarity variables with respect to the linear inequality constraints of the original optimization problem.")
    //     .def("wrtLinearInequalityConstraints", wrtLinearInequalityConstraints2, rvp, "Return the complementarity variables with respect to the linear inequality constraints of the original optimization problem.")
    //     .def("wrtNonLinearInequalityConstraints", wrtNonLinearInequalityConstraints1, rvp, "Return the complementarity variables with respect to the non-linear inequality constraints of the original optimization problem.")
    //     .def("wrtNonLinearInequalityConstraints", wrtNonLinearInequalityConstraints2, rvp, "Return the complementarity variables with respect to the non-linear inequality constraints of the original optimization problem.")
    //     ;
}
