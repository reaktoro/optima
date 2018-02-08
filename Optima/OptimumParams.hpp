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

#pragma once

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

// Forward declarations
class OptimumStructure;

/// The parameters of an optimization problem that change with more frequency.
class OptimumParams
{
public:
    /// Construct a default OptimumParams instance.
    /// @param structure The structure of the optimization problem.
    OptimumParams(const OptimumStructure& structure);

    /// Return right-hand side vector of the equality constraint \eq{Ax = b}.
    auto b() -> VectorXdRef { return m_b; }

    /// Return right-hand side vector of the equality constraint \eq{Ax = b}.
    auto b() const -> VectorXdConstRef { return m_b; }

    /// Return the lower bound values for the variables in \eq{x} bounded below.
    auto xlower() -> VectorXdRef { return m_xlower; }

    /// Return the lower bound values for the variables in \eq{x} bounded below.
    auto xlower() const -> VectorXdConstRef { return m_xlower; }

    /// Return the upper bound values for the variables in \eq{x} bounded above.
    auto xupper() -> VectorXdRef { return m_xupper; }

    /// Return the upper bound values for the variables in \eq{x} bounded above.
    auto xupper() const -> VectorXdConstRef { return m_xupper; }

    /// Return the values for the fixed variables in \eq{x}.
    auto xfixed() -> VectorXdRef { return m_xfixed; }

    /// Return the values for the fixed variables in \eq{x}.
    auto xfixed() const -> VectorXdConstRef { return m_xfixed; }

private:
    /// The right-hand side vector of the linear equality constraint \eq{Ax = b}.
    VectorXd m_b;

    /// The lower bounds of the variables \eq{x}.
    VectorXd m_xlower;

    /// The upper bounds of the variables \eq{x}.
    VectorXd m_xupper;

    /// The values of the variables in \eq{x} that are fixed.
    VectorXd m_xfixed;

};

} // namespace Optima
