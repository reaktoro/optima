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

#include "OptimumStructure.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>

namespace Optima {

OptimumStructure::OptimumStructure(Index n, Index m)
: _n(n), _m(m), _nlower(0), _nupper(0), _nfixed(0),
  _lowerpartition(indices(n)),
  _upperpartition(indices(n)),
  _fixedpartition(indices(n))
{}

auto OptimumStructure::setVariablesWithLowerBounds(IndicesConstRef inds) -> void
{
    _nlower = inds.size();
    partitionLeft(_lowerpartition, inds);
}

auto OptimumStructure::allVariablesHaveLowerBounds() -> void
{
    _nlower = _n;
    _lowerpartition = indices(_n);
}

auto OptimumStructure::setVariablesWithUpperBounds(IndicesConstRef inds) -> void
{
    _nupper = inds.size();
    partitionLeft(_upperpartition, inds);
}

auto OptimumStructure::allVariablesHaveUpperBounds() -> void
{
    _nupper = _n;
    _upperpartition = indices(_n);
}

auto OptimumStructure::setVariablesWithFixedValues(IndicesConstRef inds) -> void
{
    _nfixed = inds.size();
    partitionLeft(_fixedpartition, inds);
}

auto OptimumStructure::numVariables() const -> Index
{
    return _n;
}

auto OptimumStructure::numEqualityConstraints() const -> Index
{
    return A.rows();
}

auto OptimumStructure::variablesWithLowerBounds() const -> IndicesConstRef
{
    return _lowerpartition.tail(_nlower);
}

auto OptimumStructure::variablesWithUpperBounds() const -> IndicesConstRef
{
    return _upperpartition.tail(_nupper);
}

auto OptimumStructure::variablesWithFixedValues() const -> IndicesConstRef
{
    return _fixedpartition.tail(_nfixed);
}

auto OptimumStructure::variablesWithoutLowerBounds() const -> IndicesConstRef
{
    return _lowerpartition.head(_n - _nlower);
}

auto OptimumStructure::variablesWithoutUpperBounds() const -> IndicesConstRef
{
    return _upperpartition.head(_n - _nupper);
}

auto OptimumStructure::variablesWithoutFixedValues() const -> IndicesConstRef
{
    return _fixedpartition.head(_n - _nfixed);
}

auto OptimumStructure::orderingLowerBounds() const -> IndicesConstRef
{
    return _lowerpartition;
}

auto OptimumStructure::orderingUpperBounds() const -> IndicesConstRef
{
    return _upperpartition;
}

auto OptimumStructure::orderingFixedValues() const -> IndicesConstRef
{
    return _fixedpartition;
}

} // namespace Optima
