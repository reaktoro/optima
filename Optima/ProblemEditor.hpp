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
#include <Optima/Matrix.hpp>
#include <Optima/ObjectiveFunction.hpp>
#include <Optima/Result.hpp>

namespace Optima {


/// The definition of the optimization problem.
struct Problem
{
    Index const n;         ///< The number of primal variables *x*.
    Index const me;        ///< The number of linear and nonlinear equality constraints.
    Index const mg;        ///< The number of linear and nonlinear inequality constraints.
    MatrixConstRef Ae;     ///< The coefficient matrix \eq{A_{\mathrm{e}}} in the linear equality constraints \eq{A_{\mathrm{e}}x=b_{\mathrm{e}}}.
    MatrixConstRef Ag;     ///< The coefficient matrix \eq{A_{\mathrm{g}}} in the linear inequality constraints \eq{A_{\mathrm{g}}x\ge b_{\mathrm{g}}}.
    VectorRef be;          ///< The right-hand side vector \eq{b_{\mathrm{e}}} in the linear equality constraints \eq{A_{\mathrm{e}}x=b_{\mathrm{e}}}.
    VectorRef bg;          ///< The right-hand side vector \eq{b_{\mathrm{g}}} in the linear inequality constraints \eq{A_{\mathrm{g}}x\ge b_{\mathrm{g}}}.
    ConstraintFunction he; ///< The nonlinear equality constraint function \eq{h_{\mathrm{e}}(x)=0}.
    ConstraintFunction hg; ///< The nonlinear inequality constraint function \eq{h_{\mathrm{g}}(x)\geq0}.
    ObjectiveFunction f;   ///< The objective function \eq{f(x)} of the optimization problem.
    VectorRef xlower;      ///< The lower bounds of the variables \eq{x}.
    VectorRef xupper;      ///< The upper bounds of the variables \eq{x}.
    Matrix dgdp;           ///< The derivatives *∂g/∂p*.
    Matrix dhdp;           ///< The derivatives *∂h/∂p*.
    Matrix dbdp;           ///< The derivatives *∂b/∂p*.
};

struct State
{
    /// The variables *x* of the basic optimization problem.
    Vector x;

    /// The Lagrange multipliers *y* with respect to constraints *Ax = b* and *h(x) = 0*.
    Vector y;

    /// The instability measures of the variables *X* defined as *z = g + tr(A)yl + tr(J)yn*.
    Vector z;
};

struct StateRef
{
    /// The variables *x* of the basic optimization problem.
    VectorRef x;

    /// The Lagrange multipliers *y* with respect to constraints *Ax = b* and *h(x) = 0*.
    VectorRef y;

    /// The instability measures of the variables *X* defined as *z = g + tr(A)yl + tr(J)yn*.
    VectorRef z;

    /// The sensitivity derivatives *∂x/∂p*.
    MatrixRef dxdp;

    /// The sensitivity derivatives *∂y/∂p*.
    MatrixRef dydp;

    /// The sensitivity derivatives *∂z/∂p*.
    MatrixRef dzdp;
};

class Solver
{
public:
    Solver();

    Solver(Index n);

    Solver(MatrixConstRef Ae);

    Solver(MatrixConstRef Ae, MatrixConstRef Ag);

    /// The number of primal variables *x*.
    Index const n;

    /// The coefficient matrix \eq{A_{\mathrm{e}}} in the linear equality constraints \eq{A_{\mathrm{e}}x=b_{\mathrm{e}}}.
    MatrixConstRef Ae;

    /// The coefficient matrix \eq{A_{\mathrm{g}}} in the linear inequality constraints \eq{A_{\mathrm{g}}x\ge b_{\mathrm{g}}}.
    MatrixConstRef Ag;

    /// The right-hand side vector \eq{b_{\mathrm{e}}} in the linear equality constraints \eq{A_{\mathrm{e}}x=b_{\mathrm{e}}}.
    VectorRef be;

    /// The right-hand side vector \eq{b_{\mathrm{g}}} in the linear inequality constraints \eq{A_{\mathrm{g}}x\ge b_{\mathrm{g}}}.
    VectorRef bg;

    /// The nonlinear equality constraint function \eq{h_{\mathrm{e}}(x)=0}.
    ConstraintFunction he;

    /// The nonlinear inequality constraint function \eq{h_{\mathrm{g}}(x)\geq0}.
    ConstraintFunction hg;

    /// The objective function \eq{f(x)} of the optimization problem.
    ObjectiveFunction f;

    /// The lower bounds of the variables \eq{x}.
    VectorRef xlower;

    /// The upper bounds of the variables \eq{x}.
    VectorRef xupper;

    /// The derivatives *∂g/∂p*.
    Matrix dgdp;

    /// The derivatives *∂h/∂p*.
    Matrix dhdp;

    /// The derivatives *∂b/∂p*.
    Matrix dbdp;

    auto solve(StateRef state) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
}

// /// The definition of the optimization problem.
// class Problem
// {
// public:
//     /// Construct a default Problem instance.
//     Problem();

//     /// Construct a Problem instance with given objective and constraints.
//     /// @param objective The objective function of the optimization problem.
//     /// @param constraints The constraints of the optimization problem.
//     Problem(const ObjectiveFunction &objective, const Constraints &constraints);

//     /// Set the right-hand side vector \eq{b_{e}} of the equality constraint equation \eq{A_{e}x=b_{e}}.
//     auto setEqualityConstraintVector(VectorConstRef be) -> void;

//     /// Set the right-hand side vector \eq{b_{i}} of the equality constraint equation \eq{A_{i}x\geq b_{i}}.
//     auto setInequalityConstraintVector(VectorConstRef bi) -> void;

//     /// Set a common lower bound value for all variables in \eq{x} that have lower bounds.
//     auto setLowerBound(double val) -> void;

//     /// Set the lower bound values for all variables in \eq{x} that have lower bounds.
//     auto setLowerBounds(VectorConstRef xlower) -> void;

//     /// Set a common upper bound value for all variables in \eq{x} that have upper bounds.
//     auto setUpperBound(double val) -> void;

//     /// Set the upper bound values for all variables in \eq{x} that have upper bounds.
//     auto setUpperBounds(VectorConstRef xupper) -> void;

//     /// Set a common fixed value for all variables in \eq{x} that have fixed values.
//     auto setFixedValue(double val) -> void;

//     /// Set the fixed values of all variables in \eq{x} that have fixed values.
//     auto setFixedValues(VectorConstRef xfixed) -> void;

//     /// Return the objective function of the optimization problem.
//     auto objective() const -> const ObjectiveFunction &;

//     /// Return the constraints of the optimization problem.
//     auto constraints() const -> const Constraints &;

//     /// Return right-hand side vector \eq{b_{e}} of the equality constraint equation \eq{A_{e}x=b_{e}}.
//     auto equalityConstraintVector() const -> VectorConstRef;

//     /// Return the right-hand side vector \eq{b_{i}} of the equality constraint equation \eq{A_{i}x\geq b_{i}}.
//     auto inequalityConstraintVector() const -> VectorConstRef;

//     /// Return the lower bound values of the variables in \eq{x} that have lower bounds.
//     auto lowerBounds() const -> VectorConstRef;

//     /// Return the upper bound values of the variables in \eq{x} that have upper bounds.
//     auto upperBounds() const -> VectorConstRef;

//     /// Return the fixed values of the variables in \eq{x} that have fixed values.
//     auto fixedValues() const -> VectorConstRef;

// private:
//     /// The objective function of the optimization problem.
//     ObjectiveFunction m_objective;

//     /// The constraints of the optimization problem.
//     Constraints m_constraints;

//     /// The right-hand side vector of the linear equality constraint \eq{A_{e}x = b_{e}}.
//     Vector m_be;

//     /// The right-hand side vector of the linear inequality constraint \eq{A_{i}x\ge b_{i}}.
//     Vector m_bi;

//     /// The lower bounds of the variables \eq{x}.
//     Vector m_xlower;

//     /// The upper bounds of the variables \eq{x}.
//     Vector m_xupper;

//     /// The values of the variables in \eq{x} that are fixed.
//     Vector m_xfixed;
// };

} // namespace Optima
