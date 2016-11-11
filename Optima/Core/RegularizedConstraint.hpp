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

/// A type used to describe a system of linear constraints.
class RegularizedConstraint
{
public:
    /// Construct a default RegularizedConstraint instance.
    RegularizedConstraint();

    /// Construct a RegularizedConstraint instance.
    RegularizedConstraint(const Constraint& constraint);

    /// Destroy this RegularizedConstraint instance.
    virtual ~RegularizedConstraint();

    /// Set the 
    auto be(const Vector& be) -> void;
    auto bi(const Vector& bi) -> void;

    /// Regularize the linear constraints.
    /// @param x The state of the variables `x`.
    auto regularize(const Vector& x) -> void;

    /// Return the regularizer matrix of the coefficient matrix of the linear equality constraints.
    auto Re() const -> const Matrix&;

    /// Return the regularizer matrix of the coefficient matrix of the linear inequality constraints.
    auto Ri() const -> const Matrix&;

    /// Return the regularized coefficient matrix of the linear equality constraints corresponding to the non-basic variables only.
    /// The regularized coefficient matrix corresponding to the basic variables is `Xb = diag(xb)`.
    auto Bn() const -> const Matrix&;

    /// Return the regularized right-hand side vector of the linear equality constraints.
    auto b() const -> const Vector&;

    /// Return the indices of the basic variables.
    auto ibasic() const -> const Indices&;

    /// Return the indices of the non-basic variables.
    auto inonbasic() const -> const Indices&;

    /// Return the indices of the variables with given fixed values.
    auto ifixed() const -> const Indices&;

private:
    struct Impl;

    std::shared_ptr<Impl> pimpl;
};

} // namespace Optima
