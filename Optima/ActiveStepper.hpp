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
#include <Optima/Matrix.hpp>

namespace Optima {

// Forward declarations
class SaddlePointMatrix;
class SaddlePointVector;
class Options;

/// The arguments for the construction of a ActiveStepper object.
struct ActiveStepperInitArgs
{
    Index n;                ///< The number of primal variables *x*.
    Index m;                ///< The number of Lagrange multipliers *y*.
    MatrixConstRef A;       ///< The coefficient matrix of the linear equality constraints.
    VectorConstRef xlower;  ///< The values of the lower bounds of the variables constrained with lower bounds.
    VectorConstRef xupper;  ///< The values of the upper bounds of the variables constrained with upper bounds.
    IndicesConstRef ilower; ///< The indices of the variables with lower bounds.
    IndicesConstRef iupper; ///< The indices of the variables with upper bounds.
    IndicesConstRef ifixed; ///< The indices of the variables with fixed values.
};

/// The arguments for the decomposition calculation in a ActiveStepper object.
struct ActiveStepperDecomposeArgs
{
    VectorConstRef x; ///< The current state of the primal variables.
    VectorConstRef y; ///< The current state of the Lagrange multipliers.
    MatrixConstRef J; ///< The Jacobian of the equality constraint function.
    VectorConstRef g; ///< The gradient of the objective function.
    MatrixConstRef H; ///< The Hessian of the objective function.
};

/// The arguments for the solve calculation in a ActiveStepper object.
struct ActiveStepperSolveArgs
{
    VectorConstRef x; ///< The current state of the primal variables.
    VectorConstRef y; ///< The current state of the Lagrange multipliers.
    VectorConstRef b; ///< The right-hand side vector of the linear equality constraints.
    VectorConstRef h; ///< The value of the equality constraint function.
    VectorConstRef g; ///< The gradient of the objective function.
};

/// The solution of the step problem.
struct ActiveStepperSolution
{
    VectorRef dx; ///< The calculated step for the primal variables *x*.
    VectorRef dy; ///< The calculated step for the Lagrange multipliers *y*.
    VectorRef rx; ///< The calculated residuals of the first-order optimality conditions.
    VectorRef ry; ///< The calculated residuals of the linear/nonlinear feasibility conditions.
    VectorRef z;  ///< The calculated *unstabilities* of the variables defined as `z = g + tr(W)*y`.
};

/// The class that implements the step calculation.
class ActiveStepper
{
public:
    /// Construct a default ActiveStepper instance.
    ActiveStepper();

    /// Construct a ActiveStepper instance with given initialization data.
    explicit ActiveStepper(const ActiveStepperInitArgs& args);

    /// Construct a copy of an ActiveStepper instance.
    ActiveStepper(const ActiveStepper& other);

    /// Destroy this ActiveStepper instance.
    virtual ~ActiveStepper();

    /// Assign an ActiveStepper instance to this.
    auto operator=(ActiveStepper other) -> ActiveStepper&;

    /// Set the options for the step calculation.
    auto setOptions(const Options& options) -> void;

    /// Decompose the saddle point matrix used to compute the Newton steps for *x* and *y*.
    auto decompose(const ActiveStepperDecomposeArgs& args) -> void;

    /// Solve the saddle point problem to compute the Newton steps for *x* and *y*.
    /// @note Method ActiveStepper::decompose needs to be called first.
    auto solve(const ActiveStepperSolveArgs& args, ActiveStepperSolution sol) -> void;

    /// Return the saddle point matrix of the Newton step problem.
    auto matrix(const ActiveStepperDecomposeArgs& args) -> SaddlePointMatrix;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
