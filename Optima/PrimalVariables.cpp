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

#include "PrimalVariables.hpp"

// Optima includes
#include <Optima/Constraints.hpp>

namespace Optima {

PrimalVariables::PrimalVariables()
{}

PrimalVariables::PrimalVariables(const Constraints& constraints)
: nx(constraints.numVariables()),
  nxli(constraints.numLinearInequalityConstraints()),
  nxni(constraints.numNonLinearInequalityConstraints()),
  data(zeros(nx + nxli + nxni))
{
}

auto PrimalVariables::canonicalPrimalVariables() const -> VectorConstRef
{
    return data;
}

auto PrimalVariables::canonicalPrimalVariables() -> VectorRef
{
    return data;
}

auto PrimalVariables::originalPrimalVariables() const -> VectorConstRef
{
    return data.head(nx);
}

auto PrimalVariables::originalPrimalVariables() -> VectorRef
{
    return data.head(nx);
}

auto PrimalVariables::slackVariablesLinearInequalityConstraints() const -> VectorConstRef
{
    return data.segment(nx, nxli);
}

auto PrimalVariables::slackVariablesLinearInequalityConstraints() -> VectorRef
{
    return data.segment(nx, nxli);
}

auto PrimalVariables::slackVariablesNonLinearInequalityConstraints() const -> VectorConstRef
{
    return data.tail(nxni);
}

auto PrimalVariables::slackVariablesNonLinearInequalityConstraints() -> VectorRef
{
    return data.tail(nxni);
}

} // namespace Optima
