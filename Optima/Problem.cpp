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

#include "Problem.hpp"

// Optima includes
#include <Optima/Exception.hpp>

namespace Optima {

Problem::Problem()
{}

Problem::Problem(const ObjectiveFunction& objective, const Constraints& constraints)
: m_objective(objective), m_constraints(constraints)
{}

auto Problem::setEqualityConstraintVector(VectorConstRef be) -> void
{
    const auto me = m_constraints.equalityConstraintMatrix().rows();
    Assert(be.size() == me, "Problem::setEqualityConstraintVector(VectorConstRef)",
        "Mismatch number of equality constraint equations and size of given vector be.");
    m_be = be;
}

auto Problem::setInequalityConstraintVector(VectorConstRef bi) -> void
{
    const auto mi = m_constraints.inequalityConstraintMatrix().rows();
    Assert(bi.size() == mi, "Problem::setInequalityConstraintVector(VectorConstRef)",
        "Mismatch number of inequality constraint equations and size of given vector bi.");
    m_bi = bi;
}

auto Problem::setLowerBound(double val) -> void
{
    m_xlower.fill(val);
}

auto Problem::setLowerBounds(VectorConstRef xlower) -> void
{
    const auto nl = m_constraints.variablesWithLowerBounds().size();
    Assert(xlower.size() == nl, "Problem::setLowerBounds(VectorConstRef)",
        "Mismatch number of variables with lower bounds and given values to set in xlower.");
    m_xlower = xlower;
}

auto Problem::setUpperBound(double val) -> void
{
    m_xupper.fill(val);
}

auto Problem::setUpperBounds(VectorConstRef xupper) -> void
{
    const auto nu = m_constraints.variablesWithUpperBounds().size();
    Assert(xupper.size() == nu, "Problem::setUpperBounds(VectorConstRef)",
        "Mismatch number of variables with upper bounds and given values to set in xupper.");
    m_xupper = xupper;
}

auto Problem::setFixedValue(double val) -> void
{
    m_xfixed.fill(val);
}

auto Problem::setFixedValues(VectorConstRef xfixed) -> void
{
    const auto nf = m_constraints.variablesWithFixedValues().size();
    Assert(xfixed.size() == nf, "Problem::setFixedValues(VectorConstRef)",
        "Mismatch number of fixed variables and given values to set in xfixed.");
    m_xfixed = xfixed;
}

auto Problem::objective() const -> const ObjectiveFunction&
{
    return m_objective;
}

auto Problem::constraints() const -> const Constraints&
{
    return m_constraints;
}

auto Problem::equalityConstraintVector() const -> VectorConstRef
{
    return m_be;
}

auto Problem::inequalityConstraintVector() const -> VectorConstRef
{
    return m_bi;
}

auto Problem::lowerBounds() const -> VectorConstRef
{
    return m_xlower;
}

auto Problem::upperBounds() const -> VectorConstRef
{
    return m_xupper;
}

auto Problem::fixedValues() const -> VectorConstRef
{
    return m_xfixed;
}

} // namespace Optima
