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
#include <Optima/Stability.hpp>

namespace Optima {

// Forward declarations
class Options;

/// The arguments for the construction of a Stepper object.
struct StepperInitArgs
{
    Index n;                ///< The number of primal variables *x*.
    Index m;                ///< The number of Lagrange multipliers *y*.
    MatrixConstRef A;       ///< The coefficient matrix of the linear equality constraints.
};

/// The arguments for method Stepper::initialize.
struct StepperInitializeArgs
{
    VectorConstRef b;       ///< The right-hand side vector *b* of the linear equality constraints *Ax = b*.
    VectorConstRef xlower;  ///< The lower bounds of the primal variables.
    VectorConstRef xupper;  ///< The upper bounds of the primal variables.
    VectorRef x;            ///< The output state of the primal variables modified if there are strictly unstable variables.
    Stability& stability;   ///< The output stability state of the primal variables *x* after *strictly lower unstable* and *strictly upper unstable* are determined.
};

/// The arguments for method Stepper::decompose.
struct StepperDecomposeArgs
{
    VectorConstRef x;       ///< The current state of the primal variables.
    VectorConstRef y;       ///< The current state of the Lagrange multipliers.
    VectorConstRef g;       ///< The gradient of the objective function.
    MatrixConstRef H;       ///< The Hessian of the objective function.
    MatrixConstRef J;       ///< The Jacobian of the equality constraint function.
    VectorConstRef xlower;  ///< The lower bounds of the primal variables.
    VectorConstRef xupper;  ///< The upper bounds of the primal variables.
    Stability& stability;   ///< The output stability state of the primal variables *x*.
};

/// The arguments for method Stepper::solve.
struct StepperSolveArgs
{
    VectorConstRef x;           ///< The current state of the primal variables.
    VectorConstRef y;           ///< The current state of the Lagrange multipliers.
    VectorConstRef b;           ///< The right-hand side vector *b* of the linear equality constraints *Ax = b*.
    VectorConstRef h;           ///< The value of the equality constraint function.
    VectorConstRef g;           ///< The gradient of the objective function.
    MatrixConstRef H;           ///< The Hessian of the objective function.
    Stability const& stability; ///< The stability state of the primal variables *x*.
    VectorRef dx;               ///< The output step for the primal variables *x*.
    VectorRef dy;               ///< The output step for the Lagrange multipliers *y*.
    VectorRef rx;               ///< The output residuals of the first-order optimality conditions.
    VectorRef ry;               ///< The output residuals of the linear/nonlinear feasibility conditions.
    VectorRef z;                ///< The output *unstabilities* of the variables defined as *z = g + tr(W)y* where *W = [A; J]*.
};

/// The arguments for method Stepper::sensitivities.
struct StepperSensitivitiesArgs
{
    MatrixConstRef dgdp;        ///< The derivatives *∂g/∂p*.
    MatrixConstRef dhdp;        ///< The derivatives *∂h/∂p*.
    MatrixConstRef dbdp;        ///< The derivatives *∂b/∂p*.
    Stability const& stability; ///< The stability state of the primal variables *x*.
    MatrixRef dxdp;             ///< The output sensitivity derivatives *∂x/∂p*.
    MatrixRef dydp;             ///< The output sensitivity derivatives *∂y/∂p*.
    MatrixRef dzdp;             ///< The output sensitivity derivatives *∂z/∂p*.
};

/// The class that implements the step calculation.
class Stepper
{
public:
    /// Construct a default Stepper instance.
    Stepper();

    /// Construct a Stepper instance with given initialization data.
    explicit Stepper(StepperInitArgs args);

    /// Construct a copy of a Stepper instance.
    Stepper(const Stepper& other);

    /// Destroy this Stepper instance.
    virtual ~Stepper();

    /// Assign a Stepper instance to this.
    auto operator=(Stepper other) -> Stepper&;

    /// Set the options for the step calculation.
    auto setOptions(const Options& options) -> void;

    /// Initialize the Newton step solver.
    auto initialize(StepperInitializeArgs args) -> void;

    /// Decompose the saddle point matrix used to compute the Newton steps for *x* and *y*.
    auto decompose(StepperDecomposeArgs args) -> void;

    /// Solve the saddle point problem to compute the Newton steps for *x* and *y*.
    /// @note Method Stepper::decompose needs to be called first.
    auto solve(StepperSolveArgs args) -> void;

    /// Compute the sensitivity derivatives of the saddle point problem.
    /// @note Method Stepper::solve needs to be called first.
    auto sensitivities(StepperSensitivitiesArgs args) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
