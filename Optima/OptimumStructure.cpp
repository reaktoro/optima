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

#include "OptimumStructure.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>

namespace Optima {

OptimumStructure::OptimumStructure(Index n, Index m)
: n(n), nlower(0), nupper(0), nfixed(0),
  lowerpartition(indices(n)),
  upperpartition(indices(n)),
  fixedpartition(indices(n))
{}

auto OptimumStructure::setVariablesWithLowerBounds(IndicesConstRef inds) -> void
{
    nlower = inds.size();
    partitionLeft(lowerpartition, inds);
}

auto OptimumStructure::allVariablesHaveLowerBounds() -> void
{
    nlower = n;
    lowerpartition = indices(n);
}

auto OptimumStructure::setVariablesWithUpperBounds(IndicesConstRef inds) -> void
{
    nupper = inds.size();
    partitionLeft(upperpartition, inds);
}

auto OptimumStructure::allVariablesHaveUpperBounds() -> void
{
    nupper = n;
    upperpartition = indices(n);
}

auto OptimumStructure::setVariablesWithFixedValues(IndicesConstRef inds) -> void
{
    nfixed = inds.size();
    partitionLeft(fixedpartition, inds);
}

auto OptimumStructure::numVariables() const -> Index
{
    return n;
}

auto OptimumStructure::numEqualityConstraints() const -> Index
{
    return A.rows();
}

auto OptimumStructure::variablesWithLowerBounds() const -> IndicesConstRef
{
    return lowerpartition.head(nlower);
}

auto OptimumStructure::variablesWithUpperBounds() const -> IndicesConstRef
{
    return upperpartition.head(nupper);
}

auto OptimumStructure::variablesWithFixedValues() const -> IndicesConstRef
{
    return fixedpartition.head(nfixed);
}

auto OptimumStructure::variablesWithoutLowerBounds() const -> IndicesConstRef
{
    return lowerpartition.tail(n - nlower);
}

auto OptimumStructure::variablesWithoutUpperBounds() const -> IndicesConstRef
{
    return upperpartition.tail(n - nupper);
}

auto OptimumStructure::variablesWithoutFixedValues() const -> IndicesConstRef
{
    return fixedpartition.tail(n - nfixed);
}

auto OptimumStructure::orderingLowerBounds() const -> IndicesConstRef
{
    return lowerpartition;
}

auto OptimumStructure::orderingUpperBounds() const -> IndicesConstRef
{
    return upperpartition;
}

auto OptimumStructure::orderingFixedValues() const -> IndicesConstRef
{
    return fixedpartition;
}

} // namespace Optima
