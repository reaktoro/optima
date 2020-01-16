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
#include <any>

// Optima includes
#include <Optima/ConstraintFunction.hpp>
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>
#include <Optima/ObjectiveFunction.hpp>

namespace Optima {

// Forward declarations
class Options;
class Result;

/// The state of the solution of a basic optimization problem.
struct BasicState
{
    /// The primal variables of the basic optimization problem.
    Vector x;

    /// The Lagrange multipliers with respect to the equality constraints \eq{Ax=b} and \eq{h(x)=0}.
    Vector y;

    /// The slack variables with respect to the lower bounds of the primal variables,
    Vector z;

    /// The slack variables with respect to the upper bounds of the primal variables,
    Vector w;
};

/// The dimensions of variables and constraints in a basic optimization problem.
struct BasicDims
{
    /// The number of variables (equivalent to the dimension of vector \eq{x}).
    Index x = 0;

    /// The number of linear equality constraint equations (equivalent to the dimension of vector \eq{b}).
    Index b = 0;

    /// The number of non-linear equality constraint equations (equivalent to the dimension of vector \eq{h}).
    Index h = 0;

    /// The number of variables with lower bounds (equivalent to the dimension of vector of lower bounds \eq{x_\mathrm{l}}).
    Index xlower = 0;

    /// The number of variables with upper bounds (equivalent to the dimension of vector of upper bounds \eq{x_\mathrm{u}}).
    Index xupper = 0;

    /// The number of variables with fixed values (equivalent to the dimension of vector \eq{x_\mathrm{f}}).
    Index xfixed = 0;
};

/// The constraints in a basic optimization problem.
struct BasicConstraints
{
    /// The coefficient matrix of the linear equality constraint equations \eq{Ax=b}.
    Matrix A;

    /// The constraint function in the non-linear equality constraint equations \eq{h(x) = 0}.
    ConstraintFunction h;

    /// The indices of the variables with lower bounds.
    Indices ilower;

    /// The indices of the variables with upper bounds.
    Indices iupper;

    /// The indices of the variables with fixed values.
    Indices ifixed;
};

/// The parameters of a basic optimization problem.
struct BasicParams
{
    /// The right-hand side vector of the linear equality constraints \eq{Ax = b}.
    Vector b;

    /// The lower bounds of the variables in \eq{x} that have lower bounds.
    Vector xlower;

    /// The upper bounds of the variables \eq{x} that have upper bounds.
    Vector xupper;

    /// The values of the variables in \eq{x} that are fixed.
    Vector xfixed;

    /// The extra parameters in the problem.
    std::any extra;
};

/// The definition of a basic optimization problem.
struct BasicProblem
{
    /// The dimensions of the basic optimization problem.
    BasicDims dims;

    /// The constraints of the basic optimization problem.
    BasicConstraints constraints;

    /// The objective function of the basic optimization problem.
    ObjectiveFunction objective;
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

    /// Construct a BasicSolver instance with given optimization problem.
    BasicSolver(const BasicProblem& problem);

    /// Construct a copy of a BasicSolver instance.
    BasicSolver(const BasicSolver& other);

    /// Destroy this BasicSolver instance.
    virtual ~BasicSolver();

    /// Assign a BasicSolver instance to this.
    auto operator=(BasicSolver other) -> BasicSolver&;

    /// Set the options for the optimization calculation.
    auto setOptions(const Options& options) -> void;

    /// Solve an optimization problem.
    /// @param params The parameters for the optimization calculation.
    /// @param state[in,out] The initial guess and the final state of the optimization calculation.
    auto solve(const BasicParams& params, BasicState& state) -> Result;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
