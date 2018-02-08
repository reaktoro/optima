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

#include "OptimumParams.hpp"

// Optima includes
#include <Optima/OptimumStructure.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

OptimumParams::OptimumParams(const OptimumStructure& structure)
: m_b(structure.numEqualityConstraints()),
  m_xlower(structure.variablesWithLowerBounds().size()),
  m_xupper(structure.variablesWithUpperBounds().size()),
  m_xfixed(structure.variablesWithFixedValues().size())
{
    m_b.fill(0.0);
    m_xlower.fill(0.0);
    m_xupper.fill(0.0);
    m_xfixed.fill(0.0);
}

} // namespace Optima
