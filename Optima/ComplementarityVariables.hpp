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

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

// Forward declarations
class Constraints;

/// A type that describes the Lagrange multipliers in a canonical optimization problem.
class ComplementarityVariables
{
public:
    /// Construct a default ComplementarityVariables instance.
    ComplementarityVariables();

    /// Construct a ComplementarityVariables instance with given constraints.
    ComplementarityVariables(const Constraints& constraints);


    /// Return the complementarity variables with respect to the lower bound constraints of the canonical optimization problem.
    auto wrtCanonicalLowerBounds() const -> VectorConstRef;

    /// Return the complementarity variables with respect to the lower bound constraints of the canonical optimization problem.
    auto wrtCanonicalLowerBounds() -> VectorRef;


    /// Return the complementarity variables with respect to the upper bound constraints of the canonical optimization problem.
    auto wrtCanonicalUpperBounds() const -> VectorConstRef;

    /// Return the complementarity variables with respect to the upper bound constraints of the canonical optimization problem.
    auto wrtCanonicalUpperBounds() -> VectorRef;


    /// Return the complementarity variables with respect to the lower bound constraints of the original optimization problem.
    auto wrtLowerBounds() const -> VectorConstRef;

    /// Return the complementarity variables with respect to the lower bound constraints of the original optimization problem.
    auto wrtLowerBounds() -> VectorRef;


    /// Return the complementarity variables with respect to the upper bound constraints of the original optimization problem.
    auto wrtUpperBounds() const -> VectorConstRef;

    /// Return the complementarity variables with respect to the upper bound constraints of the original optimization problem.
    auto wrtUpperBounds() -> VectorRef;


    /// Return the complementarity variables with respect to the linear inequality constraints of the original optimization problem.
    auto wrtLinearInequalityConstraints() const -> VectorConstRef;

    /// Return the complementarity variables with respect to the linear inequality constraints of the original optimization problem.
    auto wrtLinearInequalityConstraints() -> VectorRef;


    /// Return the complementarity variables with respect to the non-linear inequality constraints of the original optimization problem.
    auto wrtNonLinearInequalityConstraints() const -> VectorConstRef;

    /// Return the complementarity variables with respect to the non-linear inequality constraints of the original optimization problem.
    auto wrtNonLinearInequalityConstraints() -> VectorRef;

private:
    /// The number of original primal variables with lower bound constraints.
    Index mxl = 0;

    /// The number of original primal variables with upper bound constraints.
    Index mxu = 0;

    /// The number of linear inequality constraints in the original optimization problem.
    Index mli = 0;

    /// The number of non-linear inequality constraints in the original optimization problem.
    Index mni = 0;

    /// The vector containing the complementarity variables of the canonical optimization problem (canonical lower bounds).
    Vector data_lower;

    /// The vector containing the complementarity variables of the canonical optimization problem (canonical upper bounds).
    Vector data_upper;
};

} // namespace Optima
