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

#include "OptimumParams.hpp"

// Optima includes
#include <Optima/Common/Exception.hpp>
#include <Optima/Core/OptimumStructure.hpp>

namespace Optima {

OptimumParams::OptimumParams(const OptimumStructure& structure)
: m_b(structure.A.rows()),
  m_xlower(structure.ilower().size()),
  m_xupper(structure.iupper().size()),
  m_xfixed(structure.ifixed().size())
{
    m_xlower.fill(0.0);
    m_xupper.fill(0.0);
    m_xfixed.fill(0.0);
}

} // namespace Optima
