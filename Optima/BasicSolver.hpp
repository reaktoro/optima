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
#include <Optima/ConstraintFunction.hpp>
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>
#include <Optima/ObjectiveFunction.hpp>
#include <Optima/Stability.hpp>

namespace Optima {

// Forward declarations
class Options;
class Result;

/// The data needed in the constructor of class BasicSolver.
struct BasicSolverInitArgs
{
    Index nx;       ///< The number of primal variables *x*.
    Index np;       ///< The number of parameter variables *p*.
    Index ny;       ///< The number of Lagrange multipliers *y* (i.e. number of rows in *A = [Ax Ap]*).
    Index nz;       ///< The number of Lagrange multipliers *z* (i.e. number of equations in *h(x, p) = 0*).
    MatrixView Ax;  ///< The coefficient matrix *Ax* of the linear equality constraints.
    MatrixView Ap;  ///< The coefficient matrix *Ap* of the linear equality constraints.
};

/// The data needed in method BasicSolver::solve.
struct BasicSolverSolveArgs
{
    ObjectiveFunction const& obj;  ///< The objective function *f(x)* of the basic optimization problem.
    ConstraintFunction const& h;   ///< The nonlinear equality constraint function *h(x, p)*.
    ConstraintFunction const& v;   ///< The nonlinear constraint function *v(x, p)*.
    VectorView b;                  ///< The right-hand side vector *b* of the linear equality constraints <em>Ax*x + Ap*p = b</em>.
    VectorView xlower;             ///< The lower bounds of the primal variables *x*.
    VectorView xupper;             ///< The upper bounds of the primal variables *x*.
    VectorView plower;             ///< The lower bounds of the parameter variables *p*.
    VectorView pupper;             ///< The upper bounds of the parameter variables *p*.
    VectorRef x;                   ///< The output primal variables *x* of the basic optimization problem.
    VectorRef p;                   ///< The output parameter variables *p* of the basic optimization problem.
    VectorRef y;                   ///< The output Lagrange multipliers *y* with respect to constraints <em>Ax*x + Ap*p = b</em>.
    VectorRef z;                   ///< The output Lagrange multipliers *z* with respect to constraints *h(x, p) = 0*.
    VectorRef s;                   ///< The output stability measures of the primal variables defined as *s = g + tr(Ax)y + tr(Jx)z*.
    Stability& stability;          ///< The output stability state of the primal variables *x*.
};

/// The arguments for method BasicSolver::sensitivities.
struct BasicSolverSensitivitiesArgs
{
    MatrixView fxw;             ///< The derivatives *∂fx/∂w*.
    MatrixView hw;              ///< The derivatives *∂h/∂w*.
    MatrixView bw;              ///< The derivatives *∂b/∂w*.
    MatrixView vw;              ///< The derivatives *∂v/∂w*.
    Stability const& stability; ///< The stability state of the primal variables *x*.
    MatrixRef xw;               ///< The output sensitivity derivatives *∂x/∂w*.
    MatrixRef pw;               ///< The output sensitivity derivatives *∂p/∂w*.
    MatrixRef yw;               ///< The output sensitivity derivatives *∂y/∂w*.
    MatrixRef zw;               ///< The output sensitivity derivatives *∂z/∂w*.
    MatrixRef sw;               ///< The output sensitivity derivatives *∂s/∂w*.
};

/// The solver for optimization problems in its basic form.
///
/// @eqc{\min_{x}f(x,p)\quad\text{subject to\ensuremath{\quad\left\{ \begin{array}{c}A_{\mathrm{x}}x+A_{\mathrm{p}}p=b\\[1mm]h(x,p)=0\\[1mm]x_{l}\leq x\leq x_{u}\end{array}\right.}}\quad\text{and}\quad v(x,p)=0}
///
class BasicSolver
{
public:
    /// Construct a BasicSolver instance with given details of the optimization problem.
    BasicSolver(BasicSolverInitArgs args);

    /// Construct a copy of a BasicSolver instance.
    BasicSolver(const BasicSolver& other);

    /// Destroy this BasicSolver instance.
    virtual ~BasicSolver();

    /// Assign a BasicSolver instance to this.
    auto operator=(BasicSolver other) -> BasicSolver&;

    /// Set the options for the optimization calculation.
    auto setOptions(const Options& options) -> void;

    /// Solve the optimization problem.
    auto solve(BasicSolverSolveArgs args) -> Result;

    /// Compute the sensitivity derivatives of the optimal solution.
    /// @note Method BasicSolver::solve needs to be called first.
    auto sensitivities(BasicSolverSensitivitiesArgs args) -> Result;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
