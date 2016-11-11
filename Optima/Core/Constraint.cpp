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

#include "Constraint.hpp"

namespace Optima {
namespace {

/// A type used to describe the status of a Constraint instance.
struct ConstraintStatus
{
    /// A type used to describe the updated status of constraint matrices and vectors.
    struct Updated
    {
        /// The boolean flag that indicates if `A` has been updated.
        bool A = false;

        /// The boolean flag that indicates if `B` has been updated.
        bool B = false;

        /// The boolean flag that indicates if `a` has been updated.
        bool a = false;

        /// The boolean flag that indicates if `b` has been updated.
        bool b = false;

        /// The boolean flag that indicates if `xlower` has been updated.
        bool xlower = false;

        /// The boolean flag that indicates if `xupper` has been updated.
        bool xupper = false;

        /// The boolean flag that indicates if `xfixed` has been updated.
        bool xfixed = false;
    };

    /// The updated status of the constraint matrices and vectors.
    Updated updated;
};

} // namespace

struct Constraint::Impl
{
    /// The number of variables.
    Index n = 0;

    /// The lower bounds of the variables. Empty if none.
    Vector xlower;

    /// The upper bounds of the variables. Empty if none.
    Vector xupper;

    /// The map with the fixed variables and its indices.
    std::map<Index, double> xfixed;

    /// The coefficient matrix of the linear equality constraints.
    Matrix A;

    /// The right-hand side vector of the linear equality constraints.
    Matrix a;

    /// The coefficient matrix of the linear inequality constraints.
    Matrix B;

    /// The right-hand side vector of the linear inequality constraints.
    Matrix b;

    /// True if matrix A only contains rational entries.
    bool rationalA = false;

    /// True if matrix B only contains rational entries.
    bool rationalB = false;

    /// The status of this Constraint instance.
    ConstraintStatus status;

    /// The regularized version of this Constraint instance.
    RegularizedConstraint reg;

    /// Construct a default Impl instance.
    Impl()
    {}

    /// Construct an Impl instance.
    Impl(Index n)
    : n(n)
    {}

    /// Set the lower bound of a variable.
    auto setxlower(Index index, double value) -> void
    {
        if(xlower.size() == 0)
            xlower.resize(n);
        Assert(index < n);
        xlower[index] = value;
        status.updated.xlower = true;
    }

    /// Set the lower bound of all variables to the same value.
    auto setxlower(double value) -> void
    {
        if(xlower.size() == 0)
            xlower.resize(n);
        xlower.fill(value);
        status.updated.xlower = true;
    }

    /// Set the upper bound of a variable.
    auto setxupper(Index index, double value) -> void
    {
        if(xupper.size() == 0)
            xupper.resize(n);
        Assert(index < n);
        xupper[index] = value;
        status.updated.xupper = true;
    }

    /// Set the upper bound of all variables to the same value.
    auto setxupper(double value) -> void
    {
        if(xupper.size() == 0)
            xupper.resize(n);
        xupper.fill(value);
        status.updated.xupper = true;
    }

    /// Set the value of a fixed variable.
    auto setxfixed(Index index, double value) -> void
    {
        Assert(index < n);
        xfixed[index] = value;
        status.updated.xfixed = true;
    }

    /// Set the linear equality constraint equations.
    auto equality(const Matrix& A, const Vector& a) -> void
    {
        Assert(A.cols() == n);
        Assert(A.rows() == a.rows());
        this->A = A;
        this->a = a;
        status.updated.A = true;
        status.updated.a = true;
    }

    auto equality(const Matrix& A) -> void
    {
        Assert(A.cols() == n);
        this->A = A;
        status.updated.A = true;
    }

    auto equality(const Vector& a) -> void
    {
        this->a = a;
        status.updated.a = true;
    }

    auto inequality(const Matrix& B, const Vector& b) -> void
    {
        Assert(B.cols() == n);
        Assert(B.rows() == b.rows());
        this->B = B;
        this->b = b;
        status.updated.B = true;
        status.updated.b = true;
    }

    auto inequality(const Matrix& B) -> void
    {
        Assert(B.cols() == n);
        this->B = B;
        status.updated.B = true;
    }

    auto inequality(const Vector& b) -> void
    {
        this->b = b;
        status.updated.b = true;
    }

    auto rationalA(bool value) -> void
    {
        rationalA = value;
    }

    auto rationalB(bool value) -> void
    {
        rationalB = value;
    }
};

Constraint::Constraint()
{

}

Constraint::Constraint(Index n)
: n(n)
{

}

Constraint::~Constraint()
{

}

auto Constraint::xlower(Index index, double value) -> void
{
    if(xlower.size() == 0)
        xlower.resize(n);
    Assert(index < n);
    xlower[index] = value;
}

auto Constraint::xlower(double value) -> void
{
    if(xlower.size() == 0)
        xlower.resize(n);
    xlower.fill(value);
}

auto Constraint::xupper(Index index, double value) -> void
{
    if(xupper.size() == 0)
        xupper.resize(n);
    Assert(index < n);
    xupper[index] = value;
}

auto Constraint::xupper(double value) -> void
{
    if(xupper.size() == 0)
        xupper.resize(n);
    xupper.fill(value);
}

auto Constraint::xfixed(Index index, double value) -> void
{

}

auto Constraint::equality(const Matrix& A, const Vector& a) -> void
{

}

auto Constraint::equality(const Matrix& A) -> void
{

}

auto Constraint::equality(const Vector& a) -> void
{

}

auto Constraint::inequality(const Matrix& B, const Vector& b) -> void
{

}

auto Constraint::inequality(const Matrix& B) -> void
{

}

auto Constraint::inequality(const Vector& b) -> void
{

}

auto Constraint::rationalA(bool value) -> void
{

}

auto Constraint::rationalB(bool value) -> void
{

}

auto Constraint::xlower() const -> const Vector&
{

}

auto Constraint::xupper() const -> const Vector&
{

}

auto Constraint::xfixed() const -> const Vector&
{

}

auto Constraint::ifixed() const -> const Indices&
{

}

auto Constraint::A() const -> const Matrix&
{

}

auto Constraint::B() const -> const Matrix&
{

}

auto Constraint::a() const -> const Vector&
{

}

auto Constraint::b() const -> const Vector&
{

}

auto Constraint::rationalA() const -> bool
{

}

auto Constraint::rationalB() const -> bool
{

}

} // namespace Optima
