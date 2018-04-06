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

/// The structure of an optimization problem that changes with less frequency.
class OptimumStructure
{
public:
    /// The coefficient matrix of the linear equality constraint \eq{Ax = b}.
    Matrix A;

public:
    /// Construct an OptimumStructure instance with equality constraints.
    /// @param n The number of variables in \eq{x} in the optimization problem.
    /// @param m The number of linear equality constraints in the optimization problem.
    OptimumStructure(Index n, Index m);

    /// Set the indices of the variables in \eq{x} with lower bounds.
    auto setVariablesWithLowerBounds(IndicesConstRef indices) -> void;

    /// Set all variables in \eq{x} with lower bounds.
    auto allVariablesHaveLowerBounds() -> void;

    /// Set the indices of the variables in \eq{x} with upper bounds.
    auto setVariablesWithUpperBounds(IndicesConstRef indices) -> void;

    /// Set all variables in \eq{x} with upper bounds.
    auto allVariablesHaveUpperBounds() -> void;

    /// Set the indices of the variables in \eq{x} with fixed values.
    auto setVariablesWithFixedValues(IndicesConstRef indices) -> void;

    /// Return the number of variables.
    auto numVariables() const -> Index;

    /// Return the number of linear equality constraints.
    auto numEqualityConstraints() const -> Index;

    /// Return the indices of the variables with lower bounds.
    auto variablesWithLowerBounds() const -> IndicesConstRef;

    /// Return the indices of the variables with upper bounds.
    auto variablesWithUpperBounds() const -> IndicesConstRef;

    /// Return the indices of the variables with fixed values.
    auto variablesWithFixedValues() const -> IndicesConstRef;

    /// Return the indices of the variables without lower bounds.
    auto variablesWithoutLowerBounds() const -> IndicesConstRef;

    /// Return the indices of the variables without upper bounds.
    auto variablesWithoutUpperBounds() const -> IndicesConstRef;

    /// Return the indices of the variables without fixed values.
    auto variablesWithoutFixedValues() const -> IndicesConstRef;

    /// Return the indices of the variables partitioned in [without, with] lower bounds.
    auto orderingLowerBounds() const -> IndicesConstRef;

    /// Return the indices of the variables partitioned in [without, with] upper bounds.
    auto orderingUpperBounds() const -> IndicesConstRef;

    /// Return the indices of the variables partitioned in [without, with] fixed values.
    auto orderingFixedValues() const -> IndicesConstRef;

private:
    /// The number of variables in the optimization problem.
    Index n;

    /// The number of linear equality constraints in the optimization problem.
    Index m;

    /// The number of variables with lower bounds.
    Index nlower;

    /// The number of variables with upper bounds.
    Index nupper;

    /// The number of variables with fixed values.
    Index nfixed;

    /// The indices of the variables partitioned in [with, without] lower bounds.
    Indices lowerpartition;

    /// The indices of the variables partitioned in [with, without] upper bounds.
    Indices upperpartition;

    /// The indices of the variables partitioned in [with, without] fixed values.
    Indices fixedpartition;
};

} // namespace Optima
