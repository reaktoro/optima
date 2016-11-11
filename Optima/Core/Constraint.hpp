// Reaktoro is a unified framework for modeling chemically reactive systems.
//
// Copyright (C) 2014-2015 Allan Leal
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
#include <Optima/Common/Index.hpp>
#include <Optima/Math/Matrix.hpp>

namespace Optima {

// Forward declarations
struct RegularizedConstraint;

/// A type used to describe a system of linear constraints.
class Constraint
{
public:
    /// Construct a default Constraint instance.
    Constraint();

    /// Construct a Constraint instance.
    /// @param n The number of variables.
    Constraint(Index n);

    /// Destroy this Constraint instance.
    virtual ~Constraint();

    /// Set the lower bound of a variable.
    /// @param index The index of the variable.
    /// @param value The value of the lower bound.
    auto xlower(Index index, double value) -> void;

    /// Set the lower bound of all variables to the same value.
    /// @param value The value of the lower bound.
    auto xlower(double value) -> void;

    /// Set the upper bound of a variable.
    /// @param index The index of the variable.
    /// @param value The value of the upper bound.
    auto xupper(Index index, double value) -> void;

    /// Set the upper bound of all variables to the same value.
    /// @param value The value of the upper bound.
    auto xupper(double value) -> void;

    /// Set the value of a fixed variable.
    /// @param index The index of the fixed variable.
    /// @param value The value of the fixed variable.
    auto xfixed(Index index, double value) -> void;

    /// Set the linear equality constraint equations.
    /// @param A The coefficient matrix of the equality equations.
    /// @param a The right-hand side vector of the equality equations.
    auto equality(const Matrix& A, const Vector& a) -> void;

    /// Set the coefficient matrix of the linear equality constraint equations.
    /// @param A The coefficient matrix of the equality equations.
    auto equality(const Matrix& A) -> void;

    /// Set the right-hand side of the linear equality constraint equations.
    /// @param a The right-hand side vector of the equality equations.
    auto equality(const Vector& a) -> void;

    /// Set the linear inequality constraint equations.
    /// @param B The coefficient matrix of the inequality equations.
    /// @param b The right-hand side vector of the inequality equations.
    auto inequality(const Matrix& B, const Vector& b) -> void;

    /// Set the coefficient matrix of the linear inequality constraint equations.
    /// @param B The coefficient matrix of the inequality equations.
    auto inequality(const Matrix& B) -> void;

    /// Set the right-hand side of the linear inequality constraint equations.
    /// @param b The right-hand side vector of the inequality equations.
    auto inequality(const Vector& b) -> void;

    /// Set `true` if the entries in `A` are rational numbers.
    auto rationalA(bool value) -> void;

    /// Set `true` if the entries in `B` are rational numbers.
    auto rationalB(bool value) -> void;

    /// Return the lower bounds of the variables.
    auto xlower() const -> const Vector&;

    /// Return the upper bounds of the variables.
    auto xupper() const -> const Vector&;

    /// Return the values of the fixed variables.
    /// Use method ifixed to access the corresponding indices of the fixed variables.
    auto xfixed() const -> const Vector&;

    /// Return the indices of the fixed variables.
    /// Use method xfixed to access the corresponding values of the fixed variables.
    auto ifixed() const -> const Indices&;

    /// Return the coefficient matrix of the linear equality constraints.
    auto A() const -> const Matrix&;

    /// Return the coefficient matrix of the linear inequality constraints.
    auto B() const -> const Matrix&;

    /// Return the right-hand side vector of the linear equality constraints.
    auto a() const -> const Vector&;

    /// Return the right-hand side vector of the linear inequality constraints.
    auto b() const -> const Vector&;

    /// Return `true` if the entries in `A` are rational numbers.
    auto rationalA() const -> bool;

    /// Return `true` if the entries in `A` are rational numbers.
    auto rationalB() const -> bool;

    /// Regularize the equality and inequality constraints.
    /// @param x The values of the variables.
    auto regularize(const Vector& x) -> RegularizedConstraint;

private:
    struct Impl;

    std::shared_ptr<Impl> pimpl;
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
struct RegularizedConstraint
{
    /// The regularizer matrix of `A` so that `R*A(orig) = [Ib An]`.
    Matrix R;

    /// The regularizer matrix of `B` so that `S*B(orig) = [Ib Bn]`.
    Matrix S;

    /// The inverse of the regularizer matrix `R`.
    Matrix invR;

    /// The inverse of the regularizer matrix `S`.
    Matrix invS;

    /// The regularized matrix `A` of the linear equality constraints corresponding to non-basic variables.
    /// The matrix `An` is defined as `R*A(orig) = [Ib An]`, where `Ib` is the
    /// identity matrix of dimension `nb`, the number of basic variables.
    Matrix An;

    /// The regularized matrix `B` of the linear inequality constraints corresponding to non-basic variables.
    /// The matrix `Bn` is defined as `R*B(orig) = [Ib Bn]`, where `Ib` is the
    /// identity matrix of dimension `nb`, the number of basic variables.
    Matrix Bn;

    /// The regularized vector `a` of the linear equality constraints.
    /// The vector `a` is defined as `a = R*a(orig)`.
    Vector a;

    /// The regularized vector `b` of the linear inequality constraints.
    /// The vector `b` is defined as `b = S*b(orig)`.
    Vector b;

    /// The lower bounds of the non-fixed variables.
    Vector xlower;

    /// The upper bounds of the non-fixed variables.
    Vector xupper;

    /// The values of the fixed variables.
    Vector xfixed;

    /// The indices of the fixed variables.
    Indices ifixed;

    /// The indices of the non-fixed variables.
    Indices inonfixed;

    /// The indices of the basic non-fixed variables.
    Indices ibasic;

    /// The indices of the non-basic non-fixed variables.
    Indices inonbasic;

    /// The indices of the zero variables.
    Indices izerovariables;

    /// The indices of the singular equality constraints.
    Indices izeroconstraints;

    /// The indices of the linearly independent rows of `A(orig)`.
    Indices iliA;

    /// The indices of the linearly independent rows of `B(orig)`.
    Indices iliB;
};

} // namespace Optima
