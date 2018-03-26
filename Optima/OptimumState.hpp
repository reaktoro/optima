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
#include <Optima/VariantMatrix.hpp>

namespace Optima {

// Forward declarations
class OptimumStructure;

/// Used to describe the state of an optimization calculation.
class OptimumState
{
public:
    /// The primal solution of the optimization problem.
    Vector x;

    /// The dual solution of the optimization problem with respect to the equality constraints.
    Vector y;

    /// The dual solution of the optimization problem with respect to the lower bound constraints.
    Vector z;

    /// The dual solution of the optimization problem with respect to the upper bound constraints.
    Vector w;

    /// The value of the objective function.
    double f;

    /// The gradient of the objective function.
    Vector g;

    /// The Hessian of the objective function.
    VariantMatrix H;

    /// Construct an OptimumState instance.
    /// @param x The primal solution of the optimization problem.
    /// @param y The dual solution of the optimization problem with respect to the equality constraints.
    /// @param z The dual solution of the optimization problem with respect to the lower bound constraints.
    /// @param w The dual solution of the optimization problem with respect to the upper bound constraints.
    /// @param g The gradient of the objective function.
    /// @param H The Hessian matrix of the objective function.
    OptimumState(const OptimumStructure& structure);
};

} // namespace Optima
