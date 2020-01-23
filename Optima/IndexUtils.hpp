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
#include <algorithm>

// Optima includes
#include <Optima/Matrix.hpp>
#include <Optima/Index.hpp>

namespace Optima {

/// Return a vector of indices with values from 0 up to a given length.
inline auto indices(Index length) -> decltype(linspace<Index>(length))
{
    return linspace<Index>(length);
}

/// Return `true` if a given index `i` is contained in vector `indices`.
inline auto contains(Index i, IndicesConstRef indices) -> bool
{
    const auto size = indices.size();
    const auto begin = indices.data();
    const auto end = begin + size;
    return std::find(begin, end, i) < end;
}

/// Partition `base` into (`group1`, `group2`) so that indices in `p` and `group1` are the same.
/// @param base The base vector to be partitioned.
/// @param p The indices in base vector to be moved to the left.
/// @return The index of the first entry in `group2`
/// @see partitionRight
inline auto partitionLeft(IndicesRef base, IndicesConstRef p) -> Index
{
    // The lambda function that returns true if variable i is in p
    auto in_group1 = [=](Index i) { return contains(i, p); };

    // The partitioning of base as (group1, group2)
    return std::partition(base.begin(), base.end(), in_group1) - base.begin();
}

/// Partition `base` into (`group1`, `group2`) so that indices in `p` and `group2` are the same.
/// @param base The base vector to be partitioned.
/// @param p The indices in base vector to be moved to the right.
/// @return The index of the first entry in `group2`
/// @see partitionLeft
inline auto partitionRight(IndicesRef base, IndicesConstRef p) -> Index
{
    // The lambda function that returns true if variable i is not in p
    auto in_group1 = [=](Index i) { return !contains(i, p); };

    // The partitioning of base as (group1, group2)
    return std::partition(base.begin(), base.end(), in_group1) - base.begin();
}

/// Partition `base` into (`group1`, `group2`) so that indices in `p` and `group1` are the same and **have the same order**.
/// @param base The base vector to be partitioned.
/// @param p The indices in base vector to be moved to the left.
/// @return The index of the first entry in `group2`
/// @see partitionRight
inline auto partitionLeftStable(IndicesRef base, IndicesConstRef p) -> Index
{
    // The lambda function that returns true if variable i is in p
    auto in_group1 = [=](Index i) { return contains(i, p); };

    // The partitioning of base as (group1, group2) and keep order of p in group1
    return std::stable_partition(base.begin(), base.end(), in_group1) - base.begin();
}

/// Partition `base` into (`group1`, `group2`) so that indices in `p` and `group2` are the same and **have the same order**.
/// @param base The base vector to be partitioned.
/// @param p The indices in base vector to be moved to the right.
/// @return The index of the first entry in `group2`
/// @see partitionLeft
inline auto partitionRightStable(IndicesRef base, IndicesConstRef p) -> Index
{
    // The lambda function that returns true if variable i is not in p
    auto in_group1 = [=](Index i) { return !contains(i, p); };

    // The partitioning of base as (group1, group2) and keep order of p in group2
    return std::stable_partition(base.begin(), base.end(), in_group1) - base.begin();
}

/// Return the indices in `indices1` that are not in `indices2`.
inline auto difference(IndicesConstRef indices1, IndicesConstRef indices2) -> Indices
{
    Indices tmp(indices1);
    const auto idx = partitionRight(tmp, indices2);
    return tmp.head(idx);
}

/// Return the indices in `indices1` that are in `indices2`.
inline auto intersect(IndicesConstRef indices1, IndicesConstRef indices2) -> Indices
{
    Indices tmp(indices1);
    const auto idx = partitionRight(tmp, indices2);
    return tmp.tail(tmp.size() - idx);
}

} // namespace Optima
