// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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
#include <algorithm>

// Optima includes
#include <Optima/Index.hpp>

namespace Optima {

/// Return a vector of indices with values from 0 up to a given length.
inline auto indices(Index length) -> decltype(Eigen::linspace<Index>(length))
{
    return Eigen::linspace<Index>(length);
}

/// Return `true` if a given index `i` is contained in vector `indices`.
inline auto contains(IndicesConstRef indices, Index i) -> bool
{
    const auto size = indices.size();
    const auto begin = indices.data();
    const auto end = begin + size;
    return std::find(begin, end, i) < end;
}

/// Partition `base` into (`group1`, `group2`) so that indices in `p` and `group1` are the same.
/// @param base The base vector to be partitioned.
/// @param p The indices in base vector to be moved to the left.
/// @see partitionRight
inline auto partitionLeft(IndicesRef base, IndicesConstRef p) -> void
{
    // The lambda function that returns true if variable i is in p
    auto in_group1 = [=](Index i) { return contains(p, i); };

    // The partitioning of base as (group1, group2)
    std::partition(base.data(), base.data() + base.size(), in_group1);
}


/// Partition `base` into (`group1`, `group2`) so that indices in `p` and `group2` are the same.
/// @param base The base vector to be partitioned.
/// @param p The indices in base vector to be moved to the right.
/// @see partitionLeft
inline auto partitionRight(IndicesRef base, IndicesConstRef p) -> void
{
    // The lambda function that returns true if variable i is not in p
    auto in_group1 = [=](Index i) { return !contains(p, i); };

    // The partitioning of base as (group1, group2)
    std::partition(base.data(), base.data() + base.size(), in_group1);
}

} // namespace Optima
