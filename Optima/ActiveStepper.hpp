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
#include <Optima/Number.hpp>

namespace Optima {

// Forward declarations
class Options;

/// The arguments for the construction of a ActiveStepper object.
struct ActiveStepperInitArgs
{
    Index n;                ///< The number of primal variables *x*.
    Index m;                ///< The number of Lagrange multipliers *y*.
    MatrixConstRef A;       ///< The coefficient matrix of the linear equality constraints.
};

/// The arguments for method ActiveStepper::initialize.
struct ActiveStepperInitializeArgs
{
    VectorConstRef xlower;  ///< The lower bounds of the primal variables.
    VectorConstRef xupper;  ///< The upper bounds of the primal variables.
};

/// The arguments for method ActiveStepper::decompose.
struct ActiveStepperDecomposeArgs
{
    VectorConstRef x;       ///< The current state of the primal variables.
    VectorConstRef y;       ///< The current state of the Lagrange multipliers.
    VectorConstRef g;       ///< The gradient of the objective function.
    MatrixConstRef H;       ///< The Hessian of the objective function.
    MatrixConstRef J;       ///< The Jacobian of the equality constraint function.
    VectorConstRef xlower;  ///< The lower bounds of the primal variables.
    VectorConstRef xupper;  ///< The upper bounds of the primal variables.
    IndicesRef iordering;   ///< The output ordering of the variables as (*stable*, *lower unstable*, *upper unstable*).
    IndexNumberRef nul;     ///< The output number of *lower unstable variables* (i.e. those active/attached at their lower bounds).
    IndexNumberRef nuu;     ///< The output number of *upper unstable variables* (i.e. those active/attached at their upper bounds).
};

/// The arguments for method ActiveStepper::solve.
struct ActiveStepperSolveArgs
{
    VectorConstRef x;          ///< The current state of the primal variables.
    VectorConstRef y;          ///< The current state of the Lagrange multipliers.
    VectorConstRef b;          ///< The right-hand side vector *b* of the linear equality constraints *Ax = b*.
    VectorConstRef h;          ///< The value of the equality constraint function.
    VectorConstRef g;          ///< The gradient of the objective function.
    VectorRef dx;              ///< The output step for the primal variables *x*.
    VectorRef dy;              ///< The output step for the Lagrange multipliers *y*.
    VectorRef rx;              ///< The output residuals of the first-order optimality conditions.
    VectorRef ry;              ///< The output residuals of the linear/nonlinear feasibility conditions.
    VectorRef z;               ///< The output *unstabilities* of the variables defined as *z = g + tr(W)y* where *W = [A; J]*.
};

/// The arguments for method ActiveStepper::sensitivities.
struct ActiveStepperSensitivitiesArgs
{
    MatrixConstRef dgdp; ///< The derivatives *∂g/∂p*.
    MatrixConstRef dhdp; ///< The derivatives *∂h/∂p*.
    MatrixConstRef dbdp; ///< The derivatives *∂b/∂p*.
    MatrixRef dxdp;      ///< The output sensitivity derivatives *∂x/∂p*.
    MatrixRef dydp;      ///< The output sensitivity derivatives *∂y/∂p*.
    MatrixRef dzdp;      ///< The output sensitivity derivatives *∂z/∂p*.
};

/// The class that implements the step calculation.
class ActiveStepper
{
public:
    /// Construct a default ActiveStepper instance.
    ActiveStepper();

    /// Construct a ActiveStepper instance with given initialization data.
    explicit ActiveStepper(ActiveStepperInitArgs args);

    /// Construct a copy of an ActiveStepper instance.
    ActiveStepper(const ActiveStepper& other);

    /// Destroy this ActiveStepper instance.
    virtual ~ActiveStepper();

    /// Assign an ActiveStepper instance to this.
    auto operator=(ActiveStepper other) -> ActiveStepper&;

    /// Set the options for the step calculation.
    auto setOptions(const Options& options) -> void;

    /// Initialize the Newton step solver.
    auto initialize(ActiveStepperInitializeArgs args) -> void;

    /// Decompose the saddle point matrix used to compute the Newton steps for *x* and *y*.
    auto decompose(ActiveStepperDecomposeArgs args) -> void;

    /// Solve the saddle point problem to compute the Newton steps for *x* and *y*.
    /// @note Method ActiveStepper::decompose needs to be called first.
    auto solve(ActiveStepperSolveArgs args) -> void;

    /// Compute the sensitivity derivatives of the saddle point problem.
    /// @note Method ActiveStepper::solve needs to be called first.
    auto sensitivities(ActiveStepperSensitivitiesArgs args) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
