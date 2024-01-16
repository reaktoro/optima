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
#include <Optima/LinearSolverOptions.hpp>
using namespace Optima;

void exportLinearSolverOptions(py::module& m)
{
    py::enum_<LinearSolverMethod>(m, "LinearSolverMethod")
        .value("Fullspace", LinearSolverMethod::Fullspace)
        .value("Nullspace", LinearSolverMethod::Nullspace)
        .value("Rangespace", LinearSolverMethod::Rangespace)
        ;

    py::class_<LinearSolverOptions>(m, "LinearSolverOptions")
        .def(py::init<>())
        .def_readwrite("method", &LinearSolverOptions::method)
        ;
}
