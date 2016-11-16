// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2016 Allan Leal
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
#include <map>

// Optima includes
#include <Optima/Common/Index.hpp>
#include <Optima/Math/Matrix.hpp>

namespace Optima {

/// A type used to describe a system of linear constraints.
struct Constraint
{
    /// The coefficient matrix of the linear equality constraint `A*x = a`.
    Matrix A;

    /// The right-hand side vector of the linear equality constraint `A*x = a`.
    Vector a;

    /// The coefficient matrix of the linear inequality constraint `B*x >= b`.
    Matrix B;

    /// The right-hand side vector of the linear equality constraint `B*x >= b`.
    Vector b;

    /// The lower bound of the primal variables `x`.
    Vector xlower;

    /// The upper bound of the primal variables `x`.
    Vector xupper;

    /// The values of the fixed variables.
    std::map<Index, double> xfixed;
};

/// A type used to describe an optimized constraint.
/// This constraint type represents a constraint
/// with no fixed variables and no linearly dependent
/// constraint equations.
struct ConstraintOptimized
{
    /// The coefficient matrix `A(opt)` of the optimized linear equality constraints `A*x = a`.
    Matrix A;

    /// The right-hand side vector `a(opt)` of the optimized linear equality constraints `A*x = a`.
    Vector a;

    /// The coefficient matrix `B(opt)` of the optimized linear inequality constraints `B*x >= b`.
    Matrix B;

    /// The right-hand side vector `b(opt)` of the optimized linear inequality constraints `B*x >= b`.
    Vector b;

    /// The lower bounds of the optimized variables `x(opt)`.
    Vector xlower;

    /// The upper bounds of the optimized variables `x(opt)`.
    Vector xupper;

    /// The indices of the linearly independent rows of original matrix `A(orig)`.
    Indices iliA;

    /// The indices of the linearly independent rows of original matrix `B(orig)`.
    Indices iliB;

    /// The indices of the original variables `x(orig)` fixed at their lower bounds.
    Indices ifixed_lower_bounds;

    /// The indices of the original variables `x(orig)` fixed at their upper bounds.
    Indices ifixed_upper_bounds;

    /// The indices of the original linear equality constraints only feasible at the lower bounds.
    Indices iconstraints_lower_bounds;

    /// The indices of the original linear equality constraints only feasible at the upper bounds.
    Indices iconstraints_upper_bounds;
};

/// A type used to describe a regularized constraint.
/// This type is used as the output of a linear equality
/// regularization calculation. It contains the matrix `R`
/// that regularizes the coefficient matrix `A` of the linear
/// equality constraint equation `A*x = a`.
///
/// Let `m` and `n` denote the number of rows and columns of
/// matrix `A`, where `m < n` (i.e., there are less equality
/// constraints than variables). If the linear equality constraints
/// are linearly independent (i.e., the rows of `A` are linearly
/// independent), then `m = rank(A)`. In many cases, matrix `A`
/// does have linear equality constraints as a result of some
/// automatic problem setup, so that `rank(A) < m`.
/// The linear equality constraints `A*x = a` can sometimes
/// have *singular constraints*. These are constraints that
/// can only be satified if the entries in `x` corresponding to
/// non-zero coefficients in the singular constraint equation
/// are zero.
///
/// Matrix `R` is such that `R*A' = [Ib An]`, where `Ib` is the identity matrix of
/// dimension `nb`, and `An` is a rectangular matrix of dimension
/// `nb`-by-`nn`, with `nb = rank(A)` and `nn = ncols(A) -
struct ConstraintCanonical
{
    /// The coefficient matrix in the regularized linear equality constraints `xp + An*xn = a`.
    Matrix An;

    /// The right-hand side vector in the regularized linear equality constraints `xp + An*xn = a`.
    Vector a;

    /// The lower bounds of the regularized variables.
    Vector xlower;

    /// The upper bounds of the regularized variables.
    Vector xupper;

    /// The indices of the basic regularized variables.
    Indices ibasic;

    /// The indices of the non-basic regularized variables.
    Indices inonbasic;

    /// The regularizer matrix of `A(opt)` so that `R*A(opt) = [Ib An]`.
    Matrix R;

    /// The regularizer matrix of `B(opt)` so that `S*B(opt) = [Ib Bn]`.
    Matrix S;

    /// The inverse of the regularizer matrix `R`.
    Matrix invR;

    /// The inverse of the regularizer matrix `S`.
    Matrix invS;
};

struct ConstraintOptimizer;
struct ConstraintRegularizer;

} // namespace Optima
