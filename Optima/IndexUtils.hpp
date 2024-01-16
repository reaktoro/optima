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
inline auto contains(Index i, IndicesView indices) -> bool
{
    const auto size = indices.size();
    const auto begin = indices.data();
    const auto end = begin + size;
    return std::find(begin, end, i) < end;
}

/// Partition `base` into (*group1*, *group2*) with *group1* formed with indices for which `predicate` is true.
/// @param base The indices to be partitioned.
/// @param predicate The predicate function that returns true if an index should be in *group1*.
/// @return The number of indices in *group1*
/// @see moveIntersectionRight
inline auto moveLeftIf(IndicesRef base, const std::function<bool(Index)>& predicate) -> Index
{
    return std::partition(base.begin(), base.end(), predicate) - base.begin();
}

/// Partition `base` into (*group1*, *group2*) with *group1* formed with indices for which `predicate` is true.
/// @param base The indices to be partitioned.
/// @param predicate The predicate function that returns true if an index should be in *group1*.
/// @return The number of indices in *group1*
/// @see moveIntersectionRight
inline auto stableMoveLeftIf(IndicesRef base, const std::function<bool(Index)>& predicate) -> Index
{
    return std::stable_partition(base.begin(), base.end(), predicate) - base.begin();
}

/// Partition `base` into (*group1*, *group2*) so that *group1* is formed with indices in `p` only.
/// @param base The indices to be partitioned.
/// @param p The indices in base vector to be moved to *group1*.
/// @return The number of indices in *group1*
/// @see moveIntersectionRight
inline auto moveIntersectionLeft(IndicesRef base, IndicesView p) -> Index
{
    if(p.size() == 0) // skip if p is empty
        return base.size();
    return moveLeftIf(base, [=](Index i) { return contains(i, p); });
}

/// Partition `base` into (*group1*, *group2*) with *group2* formed with indices for which `predicate` is true.
/// @param base The indices to be partitioned.
/// @param predicate The predicate function that returns true if an index should be in *group2*.
/// @return The number of indices in *group1*
/// @see moveIntersectionRight
inline auto moveRightIf(IndicesRef base, const std::function<bool(Index)>& predicate) -> Index
{
    return std::partition(base.begin(), base.end(), [&](Index i) { return !predicate(i); }) - base.begin();
}

/// Partition `base` into (*group1*, *group2*) with *group2* formed with indices for which `predicate` is true.
/// @param base The indices to be partitioned.
/// @param predicate The predicate function that returns true if an index should be in *group2*.
/// @return The number of indices in *group1*
/// @see moveIntersectionRight
inline auto stableMoveRightIf(IndicesRef base, const std::function<bool(Index)>& predicate) -> Index
{
    return std::stable_partition(base.begin(), base.end(), [&](Index i) { return !predicate(i); }) - base.begin();
}

/// Partition `base` into (*group1*, *group2*) so that *group2* is formed with indices in `p` only.
/// @param base The indices to be partitioned.
/// @param p The indices in base vector to be moved to *group2*.
/// @return The number of indices in *group1*
/// @see moveIntersectionLeft
inline auto moveIntersectionRight(IndicesRef base, IndicesView p) -> Index
{
    if(p.size() == 0) // skip if p is empty
        return base.size();
    return moveRightIf(base, [=](Index i) { return contains(i, p); });
}

/// Partition `base` into (*group1*, *group2*) so that indices in `p` and *group1* are the same and **have the same order**.
/// @param base The indices to be partitioned.
/// @param p The indices in base vector to be moved to the left.
/// @return The number of indices in *group1*
/// @see moveIntersectionRight
inline auto moveIntersectionLeftStable(IndicesRef base, IndicesView p) -> Index
{
    // Skip if p is empty
    if(p.size() == 0)
        return base.size();

    // The lambda function that returns true if variable i is in p
    auto in_group1 = [=](Index i) { return contains(i, p); };

    // The partitioning of base as (group1, group2) and keep order of p in group1
    return std::stable_partition(base.begin(), base.end(), in_group1) - base.begin();
}

/// Partition `base` into (*group1*, *group2*) so that indices in `p` and *group2* are the same and **have the same order**.
/// @param base The indices to be partitioned.
/// @param p The indices in base vector to be moved to the right.
/// @return The number of indices in *group1*
/// @see moveIntersectionLeft
inline auto moveIntersectionRightStable(IndicesRef base, IndicesView p) -> Index
{
    // Skip if p is empty
    if(p.size() == 0)
        return base.size();

    // The lambda function that returns true if variable i is not in p
    auto in_group1 = [=](Index i) { return !contains(i, p); };

    // The partitioning of base as (group1, group2) and keep order of p in group2
    return std::stable_partition(base.begin(), base.end(), in_group1) - base.begin();
}

/// Return the indices in `indices1` that are not in `indices2`.
inline auto difference(IndicesView indices1, IndicesView indices2) -> Indices
{
    Indices tmp(indices1);
    const auto idx = moveIntersectionRight(tmp, indices2);
    return tmp.head(idx);
}

/// Return the indices in `indices1` that are in `indices2`.
inline auto intersect(IndicesView indices1, IndicesView indices2) -> Indices
{
    Indices tmp(indices1);
    const auto idx = moveIntersectionRight(tmp, indices2);
    return tmp.tail(tmp.size() - idx);
}

/// Return the indices in `indices1` that are in `indices2`.
inline auto isIntersectionEmpty(IndicesView indices1, IndicesView indices2) -> bool
{
    for(Index i : indices1)
        if(contains(i, indices2))
            return false;
    return true;
}

} // namespace Optima
