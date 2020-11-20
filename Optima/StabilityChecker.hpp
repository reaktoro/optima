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
    Index nx;               ///< The number of primal variables *x*.
    Index np;               ///< The number of parameter variables *p*.
    Index ny;               ///< The number of Lagrange multipliers *y*.
    Index nz;               ///< The number of Lagrange multipliers *z*.
    MatrixView Ax;      ///< The coefficient matrix *Ax* of the linear equality constraints.
    MatrixView Ap;      ///< The coefficient matrix *Ap* of the linear equality constraints.
};

/// The arguments for method StabilityChecker::initialize.
struct StabilityCheckerInitializeArgs
{
    VectorView b;       ///< The right-hand side vector *b* of the linear equality constraints *Ax = b*.
    VectorView xlower;  ///< The lower bounds of the primal variables *x*.
    VectorView xupper;  ///< The upper bounds of the primal variables *x*.
    VectorView plower;  ///< The lower bounds of the parameter variables *p*.
    VectorView pupper;  ///< The upper bounds of the parameter variables *p*.
};

/// The arguments for method StabilityChecker::update.
struct StabilityCheckerUpdateArgs
{
    VectorView x;       ///< The current state of the primal variables *x*.
    VectorView y;       ///< The current state of the Lagrange multipliers *y*.
    VectorView z;       ///< The current state of the Lagrange multipliers *z*.
    VectorView fx;      ///< The evaluated gradient of the objective function *f(x, p)* with respect to *x*.
    MatrixView hx;      ///< The evaluated Jacobian of the equality constraint function *h(x, p)* with respect to *x*.
    VectorView xlower;  ///< The lower bounds of the primal variables *x*.
    VectorView xupper;  ///< The upper bounds of the primal variables *x*.
};

/// The arguments for method StabilityChecker::update.
struct StabilityCheckerUpdate2Args
{
    IndicesView jb;
    IndicesView jn;
    MatrixView Sbn;
    VectorView x;       ///< The current state of the primal variables *x*.
    VectorView fx;      ///< The evaluated gradient of the objective function *f(x, p)* with respect to *x*.
    VectorView xlower;  ///< The lower bounds of the primal variables *x*.
    VectorView xupper;  ///< The upper bounds of the primal variables *x*.
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

    /// Update the stability checker.
    /// This method should be called at the beginning of each iteration of the
    /// optimization calculation. It will detect *lower and upper unstable
    /// variables*, which are variables that are currently on their bounds and
    /// cannot be moved aways without increasing the Lagrange function.
    auto update2(StabilityCheckerUpdate2Args args) -> void;

    /// Return the current stability state of the primal variables *x*.
    auto stability() const -> const Stability&;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
