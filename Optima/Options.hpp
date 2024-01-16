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
#include <vector>

// Optima includes
#include <Optima/BacktrackSearchOptions.hpp>
#include <Optima/ConvergenceOptions.hpp>
#include <Optima/ErrorStatusOptions.hpp>
#include <Optima/LinearSolverOptions.hpp>
#include <Optima/LineSearchOptions.hpp>
#include <Optima/NewtonStepOptions.hpp>
#include <Optima/OutputterOptions.hpp>
#include <Optima/TransformFunction.hpp>

namespace Optima {

/// A type that describes the options for the output of a optimization calculation
struct OutputOptions : OutputterOptions
{
    /// The names of the primal variables `x`.
    /// Numbers will be used if not properly set (e.g., `x[0]`, `x[1]`)
    std::vector<std::string> xnames;

    /// The names of the primal slack variables `xbg` associated to linear inequality constraints.
    /// Numbers will be used if not properly set (e.g., `x[bg:0]`, `x[bg:1]`)
    std::vector<std::string> xbgnames;

    /// The names of the primal slack variables `xhg` associated to non-linear inequality constraints.
    /// Numbers will be used if not properly set (e.g., `x[hg:0]`, `x[hg:1]`)
    std::vector<std::string> xhgnames;

    /// The names of the parameter variables `p`.
    /// Numbers will be used if not properly set (e.g., `p[0]`, `p[1]`)
    std::vector<std::string> pnames;

    /// The names of the Lagrange multipliers `y`.
    /// Numbers will be used if not properly set (e.g., `y[0]`, `y[1]`)
    std::vector<std::string> ynames;

    /// The names of the Lagrange multipliers `z`.
    /// Numbers will be used if not properly set (e.g., `z[0]`, `z[1]`)
    std::vector<std::string> znames;
};

/// The options for the steepest descent step operation when needed.
struct SteepestDescentOptions
{
    /// The tolerance in the minimization calculation during the steepest descent operation.
    double tolerance = 1.0e-6;

    /// The maximum number of iterations during the minimization calculation during the steepest descent operation.
    double maxiters = 10;
};

/// A type that describes the options of a optimization calculation
class Options
{
public:
    /// The options for the output of the optimization calculations
    OutputOptions output;

    /// The maximum number of iterations in the optimization calculations.
    unsigned maxiters = 200;

    /// The options for assessing error status.
    ErrorStatusOptions errorstatus;

    /// The options for the backtrack search operation.
    BacktrackSearchOptions backtracksearch;

    /// The options for the linear search minimization operation.
    LineSearchOptions linesearch;

    /// The options for the steepest descent step operation when needed.
    SteepestDescentOptions steepestdescent;

    /// The options used for Newton step calculations.
    NewtonStepOptions newtonstep;

    /// The options used for convergence analysis.
    ConvergenceOptions convergence;
};

} // namespace Optima
