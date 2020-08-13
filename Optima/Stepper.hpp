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
    Index nx;                   ///< The number of primal variables *x*.
    Index np;                   ///< The number of parameter variables *p*.
    Index m;                    ///< The number of Lagrange multipliers *y*.
    MatrixConstRef Ax;          ///< The coefficient matrix *Ax* of the linear equality constraints.
    MatrixConstRef Ap;          ///< The coefficient matrix *Ap* of the linear equality constraints.
};

/// The arguments for method Stepper::initialize.
struct StepperInitializeArgs
{
    VectorConstRef b;           ///< The right-hand side vector *b* of the linear equality constraints *Ax*x + Ap*p = b*.
    VectorConstRef xlower;      ///< The lower bounds of the primal variables *x*.
    VectorConstRef xupper;      ///< The upper bounds of the primal variables *x*.
    VectorConstRef plower;      ///< The lower bounds of the parameter variables *p*.
    VectorConstRef pupper;      ///< The upper bounds of the parameter variables *p*.
    VectorRef x;                ///< The output state of the primal variables modified if there are strictly unstable variables.
    Stability& stability;       ///< The output stability state of the primal variables *x* after *strictly lower unstable* and *strictly upper unstable* are determined.
};

/// The arguments for method Stepper::canonicalize.
struct StepperCanonicalizeArgs
{
    VectorConstRef x;           ///< The current state of the primal variables *x*.
    VectorConstRef p;           ///< The current state of the parameter variables *p*.
    VectorConstRef y;           ///< The current state of the Lagrange multipliers *y*.
    VectorConstRef fx;          ///< The evaluated gradient of the objective function *f(x, p)* with respect to *x*.
    MatrixConstRef fxx;         ///< The evaluated Jacobian of the gradient function *fx(x, p)* with respect to *x*, i.e., the Hessian of *f(x)*.
    MatrixConstRef fxp;         ///< The evaluated Jacobian of the gradient function *fx(x, p)* with respect to *p*.
    MatrixConstRef vx;          ///< The evaluated Jacobian of the external constraint function *v(x, p)* with respect to *x*.
    MatrixConstRef vp;          ///< The evaluated Jacobian of the external constraint function *v(x, p)* with respect to *p*.
    MatrixConstRef hx;          ///< The evaluated Jacobian of the equality constraint function *h(x, p)* with respect to *x*.
    MatrixConstRef hp;          ///< The evaluated Jacobian of the equality constraint function *h(x, p)* with respect to *p*.
    VectorConstRef xlower;      ///< The lower bounds of the primal variables *x*.
    VectorConstRef xupper;      ///< The upper bounds of the primal variables *x*.
    VectorConstRef plower;      ///< The lower bounds of the parameter variables *p*.
    VectorConstRef pupper;      ///< The upper bounds of the parameter variables *p*.
    Stability& stability;       ///< The output stability state of the primal variables *x*.
};

/// The arguments for method Stepper::residuals.
struct StepperResidualsArgs
{
    VectorConstRef x;           ///< The current state of the primal variables *x*.
    VectorConstRef p;           ///< The current state of the parameter variables *p*.
    VectorConstRef y;           ///< The current state of the Lagrange multipliers *y*.
    VectorConstRef b;           ///< The right-hand side vector *b* of the linear equality constraints *Ax*x + Ap*p = b*.
    VectorConstRef h;           ///< The evaluated equality constraint function *h(x, p)*.
    VectorConstRef v;           ///< The evaluated external constraint function *v(x, p)*.
    VectorConstRef fx;          ///< The evaluated gradient of the objective function *f(x, p) with respect to *x*.
    MatrixConstRef hx;          ///< The evaluated Jacobian of the equality constraint function *h(x, p)* with respect to *x*.
    VectorRef rx;               ///< The output residuals of the first-order optimality conditions.
    VectorRef rp;               ///< The output residuals of the external constraint functions *v(x, p)*.
    VectorRef ry;               ///< The output residuals of the linear/nonlinear feasibility conditions.
    VectorRef ex;               ///< The output relative errors of the first-order optimality conditions.
    VectorRef ep;               ///< The output relative errors of the external constraint functions *v(x, p)*.
    VectorRef ey;               ///< The output relative errors of the linear/nonlinear feasibility conditions.
    VectorRef z;                ///< The output *unstabilities* of the variables defined as *z = g + tr(Wx)y* where *Wx = [Ax; Jx]*.
};

/// The arguments for method Stepper::decompose.
struct StepperDecomposeArgs
{
    VectorConstRef x;           ///< The current state of the primal variables *x*.
    VectorConstRef p;           ///< The current state of the parameter variables *p*.
    VectorConstRef y;           ///< The current state of the Lagrange multipliers *y*.
    VectorConstRef fx;          ///< The evaluated gradient of the objective function *f(x, p)* with respect to *x*.
    MatrixConstRef fxx;         ///< The evaluated Jacobian of the gradient function *fx(x, p)* with respect to *x*, i.e., the Hessian of *f(x)*.
    MatrixConstRef fxp;         ///< The evaluated Jacobian of the gradient function *fx(x, p)* with respect to *p*.
    MatrixConstRef vx;          ///< The evaluated Jacobian of the external constraint function *v(x, p)* with respect to *x*.
    MatrixConstRef vp;          ///< The evaluated Jacobian of the external constraint function *v(x, p)* with respect to *p*.
    MatrixConstRef hx;          ///< The evaluated Jacobian of the equality constraint function *h(x, p)* with respect to *x*.
    MatrixConstRef hp;          ///< The evaluated Jacobian of the equality constraint function *h(x, p)* with respect to *p*.
    VectorConstRef xlower;      ///< The lower bounds of the primal variables *x*.
    VectorConstRef xupper;      ///< The upper bounds of the primal variables *x*.
    VectorConstRef plower;      ///< The lower bounds of the parameter variables *p*.
    VectorConstRef pupper;      ///< The upper bounds of the parameter variables *p*.
    Stability const& stability; ///< The output stability state of the primal variables *x*.
};

/// The arguments for method Stepper::solve.
struct StepperSolveArgs
{
    VectorConstRef x;           ///< The current state of the primal variables *x*.
    VectorConstRef p;           ///< The current state of the parameter variables *p*.
    VectorConstRef y;           ///< The current state of the Lagrange multipliers *y*.
    VectorConstRef fx;          ///< The evaluated gradient of the objective function *f(x, p)* with respect to *x*.
    VectorConstRef b;           ///< The right-hand side vector *b* of the linear equality constraints *Ax*x + Ap*p = b*.
    VectorConstRef h;           ///< The evaluated equality constraint function *h(x, p)*.
    VectorConstRef v;           ///< The evaluated external constraint function *v(x, p)*.
    Stability const& stability; ///< The stability state of the primal variables *x*.
    VectorRef dx;               ///< The output step for the primal variables *x*.
    VectorRef dp;               ///< The output step for the parameter variables *p*.
    VectorRef dy;               ///< The output step for the Lagrange multipliers *y*.
};

/// The arguments for method Stepper::sensitivities.
struct StepperSensitivitiesArgs
{
    MatrixConstRef fxw;         ///< The derivatives *∂fx/∂w*.
    MatrixConstRef hw;          ///< The derivatives *∂h/∂w*.
    MatrixConstRef bw;          ///< The derivatives *∂b/∂w*.
    MatrixConstRef vw;          ///< The derivatives *∂v/∂w*.
    Stability const& stability; ///< The stability state of the primal variables *x*.
    MatrixRef xw;               ///< The output sensitivity derivatives *∂x/∂w*.
    MatrixRef pw;               ///< The output sensitivity derivatives *∂p/∂w*.
    MatrixRef yw;               ///< The output sensitivity derivatives *∂y/∂w*.
    MatrixRef zw;               ///< The output sensitivity derivatives *∂z/∂w*.
};

/// The arguments for method Stepper::steepestDescent.
struct StepperSteepestDescentLagrangeArgs
{
    VectorConstRef x;           ///< The current state of the primal variables *x*.
    VectorConstRef p;           ///< The current state of the parameter variables *p*.
    VectorConstRef y;           ///< The current state of the Lagrange multipliers *y*.
    VectorConstRef fx;          ///< The evaluated gradient of the objective function *f(x, p)* with respect to *x*.
    VectorConstRef b;           ///< The right-hand side vector *b* of the linear equality constraints *Ax*x + Ap*p = b*.
    VectorConstRef h;           ///< The evaluated equality constraint function *h(x, p)*.
    VectorConstRef v;           ///< The evaluated external constraint function *v(x, p)*.
    VectorRef dx;               ///< The output steepest descent direction for the primal variables *x*.
    VectorRef dp;               ///< The output steepest descent direction for the parameter variables *p*.
    VectorRef dy;               ///< The output steepest descent direction for the Lagrange multipliers *y*.
};

/// The arguments for method Stepper::steepestDescentError.
struct StepperSteepestDescentErrorArgs
{
    VectorConstRef x;           ///< The current state of the primal variables *x*.
    VectorConstRef p;           ///< The current state of the parameter variables *p*.
    VectorConstRef y;           ///< The current state of the Lagrange multipliers *y*.
    VectorConstRef fx;          ///< The evaluated gradient of the objective function *f(x, p)* with respect to *x*.
    MatrixConstRef fxx;         ///< The evaluated Jacobian of the gradient function *fx(x, p)* with respect to *x*, i.e., the Hessian of *f(x)*.
    MatrixConstRef fxp;         ///< The evaluated Jacobian of the gradient function *fx(x, p)* with respect to *p*.
    VectorConstRef b;           ///< The right-hand side vector *b* of the linear equality constraints *Ax*x + Ap*p = b*.
    VectorConstRef h;           ///< The evaluated equality constraint function *h(x, p)*.
    MatrixConstRef hx;          ///< The evaluated Jacobian of the equality constraint function *h(x, p)* with respect to *x*.
    MatrixConstRef hp;          ///< The evaluated Jacobian of the equality constraint function *h(x, p)* with respect to *p*.
    VectorConstRef v;           ///< The evaluated external constraint function *v(x, p)*.
    MatrixConstRef vx;          ///< The evaluated Jacobian of the external constraint function *v(x, p)* with respect to *x*.
    MatrixConstRef vp;          ///< The evaluated Jacobian of the external constraint function *v(x, p)* with respect to *p*.
    VectorRef dx;               ///< The output steepest descent direction for the primal variables *x*.
    VectorRef dp;               ///< The output steepest descent direction for the parameter variables *x*.
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

    /// Compute the steepest descent direction with respect to Lagrange function.
    /// @note Ensure method @ref canonicalize is called first.
    auto steepestDescentLagrange(StepperSteepestDescentLagrangeArgs args) -> void;

    /// Compute the steepest descent direction with respect to error function.
    /// @note Ensure method @ref canonicalize is called first.
    auto steepestDescentError(StepperSteepestDescentErrorArgs args) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
