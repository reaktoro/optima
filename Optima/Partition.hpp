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

/// Used to describe the partition of the variables into *free* and *fixed* variables.
class Partition
{
public:
    /// Construct a default Partition instance.
    Partition();

    /// Construct a Partition instance with given number of variables.
    /// @param n The number of variables
    Partition(Index n);

    /// Set the indices of the free variables, with the remaining variables becoming fixed.
    auto setFreeVariables(IndicesConstRef indices) -> void;

    /// Set the indices of the fixed variables, with the remaining variables becoming free.
    auto setFixedVariables(IndicesConstRef indices) -> void;

    /// Return the number of variables.
    auto numVariables() const -> Index;

    /// Return the number of free variables.
    auto numFreeVariables() const -> Index;

    /// Return the number of fixed variables.
    auto numFixedVariables() const -> Index;

    /// Return the indices of the free variables.
    auto freeVariables() const -> IndicesConstRef;

    /// Return the indices of the fixed variables.
    auto fixedVariables() const -> IndicesConstRef;

    /// Return the ordering of the variables partitioned into (free, fixed) variables.
    auto ordering() const -> IndicesConstRef;

private:
    /// The ordering of the variables as (free, fixed).
    Indices _ordering;

    /// The number of fixed variables.
    Index _nf;
};

} // namespace Optima
