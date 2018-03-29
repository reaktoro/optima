// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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
#include <Optima/Matrix.hpp>
#include <Optima/Objective.hpp>

namespace Optima {

// Forward declarations
class OptimumStructure;

/// The parameters of an optimization problem that change with more frequency.
class OptimumParams
{
public:
    /// The right-hand side vector of the linear equality constraint \eq{Ax = b}.
    Vector b;

    /// The lower bounds of the variables in \eq{x} that have lower bounds.
    Vector xlower;

    /// The upper bounds of the variables \eq{x} that have upper bounds.
    Vector xupper;

    /// The values of the variables in \eq{x} that are fixed.
    Vector xfixed;

    /// The objective function of the optimization calculation.
    ObjectiveFunction objective;

    /// Construct a default OptimumParams instance.
    /// @param structure The structure of the optimization problem.
    OptimumParams(const OptimumStructure& structure);
};

} // namespace Optima
