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
#include <Optima/Result.hpp>
using namespace Optima;

void exportResult(py::module& m)
{
    py::class_<Result>(m, "Result")
        .def(py::init<>())
        .def_readwrite("succeeded", &Result::succeeded)
        .def_readwrite("failure_reason", &Result::failure_reason)
        .def_readwrite("iterations", &Result::iterations)
        .def_readwrite("error", &Result::error)
        .def_readwrite("error_optimality", &Result::error_optimality)
        .def_readwrite("error_feasibility", &Result::error_feasibility)
        .def_readwrite("num_objective_evals", &Result::num_objective_evals)
        .def_readwrite("num_objective_evals_f", &Result::num_objective_evals_f)
        .def_readwrite("num_objective_evals_fx", &Result::num_objective_evals_fx)
        .def_readwrite("num_objective_evals_fxx", &Result::num_objective_evals_fxx)
        .def_readwrite("num_objective_evals_fxp", &Result::num_objective_evals_fxp)
        .def_readwrite("time", &Result::time)
        .def_readwrite("time_objective_evals", &Result::time_objective_evals)
        .def_readwrite("time_objective_evals_f", &Result::time_objective_evals_f)
        .def_readwrite("time_objective_evals_fx", &Result::time_objective_evals_fx)
        .def_readwrite("time_objective_evals_fxx", &Result::time_objective_evals_fxx)
        .def_readwrite("time_objective_evals_fxp", &Result::time_objective_evals_fxp)
        .def_readwrite("time_constraint_evals", &Result::time_constraint_evals)
        .def_readwrite("time_linear_systems", &Result::time_linear_systems)
        .def_readwrite("time_sensitivities", &Result::time_sensitivities)
        .def(py::self += py::self, py::return_value_policy::reference_internal)
        ;
}
