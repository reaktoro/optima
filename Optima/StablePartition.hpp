// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
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
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to represent the indices of stable and unstable variables.
class StablePartition
{
public:
    /// Construct a default StablePartition instance.
    /// @param nx The number of variables *x*.
    StablePartition(Index nx);

    /// Construct a copy of a StablePartition instance.
    StablePartition(const StablePartition& other);

    /// Destroy this StablePartition instance.
    virtual ~StablePartition();

    /// Assign a StablePartition instance to this.
    auto operator=(StablePartition other) -> StablePartition&;

    /// Set the indices of the stable variables in *x*.
    auto setStable(IndicesView js) -> void;

    /// Set the indices of the unstable variables in *x*.
    auto setUnstable(IndicesView ju) -> void;

    /// Return the indices of the stable variables in *x*.
    auto stable() const -> IndicesView;

    /// Return the indices of the unstable variables in *x*.
    auto unstable() const -> IndicesView;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
