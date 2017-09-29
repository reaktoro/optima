// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2017 Allan Leal
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

namespace Optima {

OptimumStructure::OptimumStructure(Index n)
: n(n), m_nlower(0), m_nupper(0), m_nfixed(0),
  m_lowerpartition(VectorXi::LinSpaced(n, 0, n - 1)),
  m_upperpartition(VectorXi::LinSpaced(n, 0, n - 1)),
  m_fixedpartition(VectorXi::LinSpaced(n, 0, n - 1))
{

}
auto OptimumStructure::withLowerBounds(VectorXiConstRef indices) -> void
{
    m_nlower = indices.size();
    m_lowerpartition.setLinSpaced(n, 0, n - 1);
    m_lowerpartition.tail(m_nlower).swap(m_lowerpartition(indices));
}

auto OptimumStructure::withLowerBounds() -> void
{
    m_nlower = n;
    m_lowerpartition.setLinSpaced(n, 0, n - 1);
}

auto OptimumStructure::withUpperBounds(VectorXiConstRef indices) -> void
{
    m_nupper = indices.size();
    m_upperpartition.setLinSpaced(n, 0, n - 1);
    m_upperpartition.tail(m_nupper).swap(m_upperpartition(indices));
}

auto OptimumStructure::withUpperBounds() -> void
{
    m_nupper = n;
    m_upperpartition.setLinSpaced(n, 0, n - 1);
}

auto OptimumStructure::withFixedValues(VectorXiConstRef indices) -> void
{
    m_nfixed = indices.size();
    m_fixedpartition.setLinSpaced(n, 0, n - 1);
    m_fixedpartition.tail(m_nfixed).swap(m_fixedpartition(indices));
}

} // namespace Optima
