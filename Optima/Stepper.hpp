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

/// The arguments for method Stepper::canonicalize.
struct StepperCanonicalizeArgs
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

/// The arguments for method Stepper::residuals.
struct StepperResidualsArgs
{
    VectorConstRef x;           ///< The current state of the primal variables.
    VectorConstRef y;           ///< The current state of the Lagrange multipliers.
    VectorConstRef b;           ///< The right-hand side vector *b* of the linear equality constraints *Ax = b*.
    VectorConstRef h;           ///< The value of the equality constraint function.
    VectorConstRef g;           ///< The gradient of the objective function.
    VectorRef rx;               ///< The output residuals of the first-order optimality conditions.
    VectorRef ry;               ///< The output residuals of the linear/nonlinear feasibility conditions.
    VectorRef ex;               ///< The output relative errors of the first-order optimality conditions.
    VectorRef ey;               ///< The output relative errors of the linear/nonlinear feasibility conditions.
    VectorRef z;                ///< The output *unstabilities* of the variables defined as *z = g + tr(W)y* where *W = [A; J]*.
};

/// The arguments for method Stepper::decompose.
struct StepperDecomposeArgs
{
    VectorConstRef x;            ///< The current state of the primal variables.
    VectorConstRef y;            ///< The current state of the Lagrange multipliers.
    VectorConstRef g;            ///< The gradient of the objective function.
    MatrixConstRef H;            ///< The Hessian of the objective function.
    MatrixConstRef J;            ///< The Jacobian of the equality constraint function.
    VectorConstRef xlower;       ///< The lower bounds of the primal variables.
    VectorConstRef xupper;       ///< The upper bounds of the primal variables.
    Stability const& stability;  ///< The output stability state of the primal variables *x*.
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

/// The arguments for method Stepper::steepestDescent.
struct StepperSteepestDescentArgs
{
    VectorConstRef x;           ///< The current state of the primal variables.
    VectorConstRef y;           ///< The current state of the Lagrange multipliers.
    VectorConstRef b;           ///< The right-hand side vector *b* of the linear equality constraints *Ax = b*.
    VectorConstRef h;           ///< The value of the equality constraint function.
    VectorConstRef g;           ///< The gradient of the objective function.
    VectorRef dx;               ///< The output steepest descent direction for the primal variables *x*.
    VectorRef dy;               ///< The output steepest descent direction for the Lagrange multipliers *y*.
};

/// The return type of method Stepper::info.
struct StepperInfo
{
    /// The indices of the basic variables.
    IndicesConstRef jb;

    /// The indices of the non-basic variables.
    IndicesConstRef jn;

    /// The canonicalization matrix *R* of *W = [A; J]*.
    MatrixConstRef R;

    /// The matrix *S* in the canonical form of *W = [A; J]*.
    MatrixConstRef S;

    /// The permutation matrix *Q* in the canonical form of *W = [A; J]*.
    IndicesConstRef Q;
};

/// The class that implements the step calculation.
class Stepper
{
public:
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

    /// Canonicalize matrix *W = [A; J]* in the Newton step problem.
    /// @note Ensure method @ref initialize is called first.
    auto canonicalize(StepperCanonicalizeArgs args) -> void;

    /// Calculate the current optimality and feasibility residuals.
    /// @note Ensure method @ref canonicalize is called first.
    auto residuals(StepperResidualsArgs args) -> void;

    /// Decompose the saddle point matrix in the Newton step problem.
    /// @note Ensure method @ref canonicalize is called first.
    auto decompose(StepperDecomposeArgs args) -> void;

    /// Solve the saddle point problem in the Newton step problem.
    /// @note Ensure method @ref decompose is called first.
    auto solve(StepperSolveArgs args) -> void;

    /// Compute the sensitivity derivatives of the saddle point problem.
    /// @note Ensure method @ref solve is called first.
    auto sensitivities(StepperSensitivitiesArgs args) -> void;

    /// Compute the steepest descent direction vectors for *x* and *y*.
    /// @note Ensure method @ref canonicalize is called first.
    auto steepestDescent(StepperSteepestDescentArgs args) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
