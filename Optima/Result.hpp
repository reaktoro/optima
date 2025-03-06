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

#pragma once

// C++ includes
#include <string>

// Optima includes
#include <Optima/Index.hpp>

namespace Optima {

/// A type that describes the result of an optimization calculation.
class Result
{
public:
    /// The flag that indicates if the optimization calculation converged.
    bool succeeded = false;

    /// The reason for the failure in the optimization calculation.
    std::string failure_reason;

    /// The flag that indicates if the failure is fatal.
    bool failure_fatal = false;

    /// The number of iterations in the optimization calculation.
    Index iterations = 0;

    /// The final residual error of the optimization calculation.
    double error = 0;

    /// The final optimality error of the optimization calculation.
    double error_optimality = 0;

    /// The final feasibility error of the optimization calculation.
    double error_feasibility = 0;

    /// The number of evaluations of the objective function in the optimization calculation.
    Index num_objective_evals = 0;

    /// The number of evaluations of *f(x, p)* in the optimization calculation.
    Index num_objective_evals_f = 0;

    /// The number of evaluations of *fx(x, p)* in the optimization calculation.
    Index num_objective_evals_fx = 0;

    /// The number of evaluations of *fxx(x, p)* in the optimization calculation.
    Index num_objective_evals_fxx = 0;

    /// The number of evaluations of *fxp(x, p)* in the optimization calculation.
    Index num_objective_evals_fxp = 0;

    /// The wall time spent for the optimization calculation (in unit of s).
    double time = 0;

    /// The wall time spent for all objective evaluations (in unit of s).
    double time_objective_evals = 0;

    /// The wall time spent for evaluating just *f(x, p)* (in unit of s).
    double time_objective_evals_f = 0;

    /// The wall time spent for evaluating just *fx(x, p)* (in unit of s).
    double time_objective_evals_fx = 0;

    /// The wall time spent for evaluating just *fxx(x, p)* (in unit of s).
    double time_objective_evals_fxx = 0;

    /// The wall time spent for evaluating just *fxp(x, p)* (in unit of s).
    double time_objective_evals_fxp = 0;

    /// The wall time spent for all contraint evaluations (in unit of s).
    double time_constraint_evals = 0;

    /// The wall time spent for all linear system solutions (in unit of s).
    double time_linear_systems = 0;

    /// The wall time spent for computing the sensitivity derivatives (in unit of s).
    double time_sensitivities = 0;

    /// Update this Result instance with another by addition.
    auto operator+=(const Result& other) -> Result&;
};

} // namespace Optima
