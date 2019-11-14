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

#pragma once

// Optima includes
#include <Optima/Index.hpp>

namespace Optima {

/// A type that describes the result of an optimization calculation.
class Result
{
public:
    /// The flag that indicates if the optimization calculation converged.
    bool succeeded = false;

    /// The number of iterations in the optimization calculation.
    Index iterations = 0;

    /// The number of evaluations of the objective function in the optimization calculation.
    Index num_objective_evals = 0;

    /// The convergence rate of the optimization calculation near the solution.
    double convergence_rate = 0;

    /// The final residual error of the optimization calculation.
    double error = 0;

    /// The final optimality error of the optimization calculation.
    double error_optimality = 0;

    /// The final feasibility error of the optimization calculation.
    double error_feasibility = 0;

    /// The final complementarity error (lower bounds) of the optimization calculation.
    double error_complementarity_lower = 0;

    /// The final complementarity error (upper bounds) of the optimization calculation.
    double error_complementarity_upper = 0;

    /// The wall time spent for the optimization calculation (in units of s).
    double time = 0;

    /// The wall time spent for all objective evaluations (in units of s).
    double time_objective_evals = 0;

    /// The wall time spent for all contraint evaluations (in units of s).
    double time_constraint_evals = 0;

    /// The wall time spent for all linear system solutions (in units of s).
    double time_linear_systems = 0;

    /// Update this Result instance with another by addition.
    auto operator+=(const Result& other) -> Result&;
};

} // namespace Optima
