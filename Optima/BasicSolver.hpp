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
#include <Optima/Number.hpp>
#include <Optima/Matrix.hpp>
#include <Optima/ObjectiveFunction.hpp>

namespace Optima {

// Forward declarations
class Options;
class Result;

/// The data needed in the constructor of class BasicSolver.
struct BasicSolverInitArgs
{
    Index n;          ///< The number of primal variables *x*.
    Index m;          ///< The number of linear and nonlinear equality constraints in *Ax = b* and *h(x) = 0*.
    MatrixConstRef A; ///< The coefficient matrix *A* of the linear equality constraints.
};

/// The data needed in method BasicSolver::solve.
struct BasicSolverSolveArgs
{
    ObjectiveFunction const& obj;  ///< The objective function *f(x)* of the basic optimization problem.
    ConstraintFunction const& h;   ///< The nonlinear equality constraint function *h(x)*.
    VectorConstRef b;              ///< The right-hand side vector *b* of the linear equality constraints *Ax = b*.
    VectorConstRef xlower;         ///< The lower bounds of the primal variables.
    VectorConstRef xupper;         ///< The upper bounds of the primal variables.
    VectorRef x;                   ///< The output primal variables *x* of the basic optimization problem.
    VectorRef y;                   ///< The output Lagrange multipliers *y* with respect to constraints *Ax = b* and *h(x) = 0*.
    VectorRef z;                   ///< The output instability measures of the primal variables defined as *z = g + tr(A)yl + tr(J)yn*.
    IndicesRef iordering;          ///< The output ordering of the variables as (*stable*, *lower unstable*, *upper unstable*).
    IndexNumberRef nul;            ///< The output number of *lower unstable variables* (i.e. those active/attached at their lower bounds).
    IndexNumberRef nuu;            ///< The output number of *upper unstable variables* (i.e. those active/attached at their upper bounds).
};

/// The arguments for method BasicSolver::sensitivities.
struct BasicSolverSensitivitiesArgs
{
    MatrixConstRef dgdp; ///< The derivatives *∂g/∂p*.
    MatrixConstRef dhdp; ///< The derivatives *∂h/∂p*.
    MatrixConstRef dbdp; ///< The derivatives *∂b/∂p*.
    MatrixRef dxdp;      ///< The output sensitivity derivatives *∂x/∂p*.
    MatrixRef dydp;      ///< The output sensitivity derivatives *∂y/∂p*.
    MatrixRef dzdp;      ///< The output sensitivity derivatives *∂z/∂p*.
};

/// The solver for optimization problems in its basic form.
///
/// @eqc{\min_{x}f(x)\quad\text{subject to\ensuremath{\quad\left\{ \begin{array}{c}Ax=b\\h(x)=0\\x_{l}\leq x\leq x_{u}\end{array}\right.}}}
///
class BasicSolver
{
public:
    /// Construct a default BasicSolver instance.
    BasicSolver();

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
