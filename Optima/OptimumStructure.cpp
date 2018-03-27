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
#include <Optima/Matrix.hpp>

namespace Optima {

OptimumStructure::OptimumStructure(Index n)
: OptimumStructure(n, 0)
{}

OptimumStructure::OptimumStructure(Index n, Index m)
: _n(n), _m(m), _A(Eigen::zeros(_m, _n)),
  _nlower(0), _nupper(0), _nfixed(0),
  _lowerpartition(Eigen::linspace<int>(n)),
  _upperpartition(Eigen::linspace<int>(n)),
  _fixedpartition(Eigen::linspace<int>(n)),
  _structure_hessian_matrix(MatrixStructure::Dense)
{}

OptimumStructure::OptimumStructure(MatrixConstRef A)
: OptimumStructure(A.cols(), A.rows())
{
    _A = A;
}

auto OptimumStructure::setVariablesWithLowerBounds(VectorXiConstRef indices) -> void
{
    _nlower = indices.size();
    _lowerpartition.setLinSpaced(_n, 0, _n - 1);
    _lowerpartition.tail(_nlower).swap(_lowerpartition(indices));
}

auto OptimumStructure::allVariablesHaveLowerBounds() -> void
{
    _nlower = _n;
    _lowerpartition.setLinSpaced(_n, 0, _n - 1);
}

auto OptimumStructure::setVariablesWithUpperBounds(VectorXiConstRef indices) -> void
{
    _nupper = indices.size();
    _upperpartition.setLinSpaced(_n, 0, _n - 1);
    _upperpartition.tail(_nupper).swap(_upperpartition(indices));
}

auto OptimumStructure::allVariablesHaveUpperBounds() -> void
{
    _nupper = _n;
    _upperpartition.setLinSpaced(_n, 0, _n - 1);
}

auto OptimumStructure::setVariablesWithFixedValues(VectorXiConstRef indices) -> void
{
    _nfixed = indices.size();
    _fixedpartition.setLinSpaced(_n, 0, _n - 1);
    _fixedpartition.tail(_nfixed).swap(_fixedpartition(indices));
}

} // namespace Optima
