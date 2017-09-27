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
#include <Optima/Common/Index.hpp>
#include <Optima/Math/Matrix.hpp>

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

//    /// Set the indices and values of the variables in \eq{x} that are fixed.
//    /// @param indices The indices of the fixed variables.
//    /// @param values The values of the fixed variables.
//    auto fix(VectorXiConstRef indices, VectorXdConstRef values) -> void;
//
//    /// Set the indices and minimum values of the variables in \eq{x} with lower bounds.
//    /// @param indices The indices of the variables with lower bounds.
//    /// @param values The minimum values of the variables with lower bounds.
//    auto lowerbounds(VectorXiConstRef indices, VectorXdConstRef values) -> void;
//
//    /// Set the indices and maximum values of the variables in \eq{x} with upper bounds.
//    /// @param indices The indices of the variables with upper bounds.
//    /// @param values The maximum values of the variables with upper bounds.
//    auto upperbounds(VectorXiConstRef indices, VectorXdConstRef values) -> void;

    /// Return right-hand side vector of the equality constraint \eq{Ax = b}.
    auto b() -> VectorXdRef { return m_b; }

    /// Return right-hand side vector of the equality constraint \eq{Ax = b}.
    auto b() const -> VectorXdConstRef { return m_b; }

    /// Return the lower bounds of the variables \eq{x}.
    auto xlower() -> VectorXdRef { return m_xlower; }

    /// Return the lower bounds of the variables \eq{x}.
    auto xlower() const -> VectorXdConstRef { return m_xlower; }

    /// Return the upper bounds of the variables \eq{x}.
    auto xupper() -> VectorXdRef { return m_xupper; }

    /// Return the upper bounds of the variables \eq{x}.
    auto xupper() const -> VectorXdConstRef { return m_xupper; }

    /// Return the values of the fixed variables in \eq{x}.
    auto xfixed() -> VectorXdRef { return m_xfixed; }

    /// Return the values of the fixed variables in \eq{x}.
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
