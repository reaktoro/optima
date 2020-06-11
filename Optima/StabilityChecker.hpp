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

// C++ includes
#include <memory>

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Stability.hpp>

namespace Optima {

/// The arguments for the construction of a StabilityChecker object.
struct StabilityCheckerInitArgs
{
    Index n;                ///< The number of primal variables *x*.
    Index m;                ///< The number of Lagrange multipliers *y*.
    MatrixConstRef A;       ///< The coefficient matrix of the linear equality constraints.
};

/// The arguments for method StabilityChecker::initialize.
struct StabilityCheckerInitializeArgs
{
    MatrixConstRef A;       ///< The coefficient matrix of the linear equality constraints.
    VectorConstRef b;       ///< The right-hand side vector *b* of the linear equality constraints *Ax = b*.
    VectorConstRef xlower;  ///< The lower bounds of the primal variables.
    VectorConstRef xupper;  ///< The upper bounds of the primal variables.
};

/// The arguments for method StabilityChecker::update.
struct StabilityCheckerUpdateArgs
{
    MatrixConstRef W;       ///< The coefficient matrix W = [A; J] of the linear/nonlinear equality constraints.
    VectorConstRef x;       ///< The current state of the primal variables.
    VectorConstRef y;       ///< The current state of the Lagrange multipliers.
    VectorConstRef g;       ///< The gradient of the objective function.
    VectorConstRef xlower;  ///< The lower bounds of the primal variables.
    VectorConstRef xupper;  ///< The upper bounds of the primal variables.
};

/// Used for checking the stability of the variables with respect to bounds.
class StabilityChecker
{
public:
    /// Construct a default StabilityChecker instance.
    StabilityChecker();

    /// Construct a StabilityChecker instance with given initialization data.
    explicit StabilityChecker(StabilityCheckerInitArgs args);

    /// Construct a copy of a StabilityChecker instance.
    StabilityChecker(const StabilityChecker& other);

    /// Destroy this StabilityChecker instance.
    virtual ~StabilityChecker();

    /// Assign a StabilityChecker instance to this.
    auto operator=(StabilityChecker other) -> StabilityChecker&;

    /// Initialize the stability checker.
    /// This method should be called at the beginning of the optimization
    /// calculation. It will detect *strictly unstable variables*, which are
    /// variables for which there are no feasible values that can
    /// simultaneously attain both the lower/upper bounds and the linear
    /// equality constraints.
    auto initialize(StabilityCheckerInitializeArgs args) -> void;

    /// Update the stability checker.
    /// This method should be called at the beginning of each iteration of the
    /// optimization calculation. It will detect *lower and upper unstable
    /// variables*, which are variables that are currently on their bounds and
    /// cannot be moved aways without increasing the Lagrange function.
    auto update(StabilityCheckerUpdateArgs args) -> void;

    /// Return the current stability state of the primal variables *x*.
    auto stability() const -> const Stability&;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
