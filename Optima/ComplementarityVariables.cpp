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

#include "ComplementarityVariables.hpp"

// Optima includes
#include <Optima/Constraints.hpp>

namespace Optima {

ComplementarityVariables::ComplementarityVariables()
{}

ComplementarityVariables::ComplementarityVariables(const Constraints& constraints)
: mxl(constraints.variablesWithLowerBounds().size()),
  mxu(constraints.variablesWithUpperBounds().size()),
  mli(constraints.numLinearInequalityConstraints()),
  mni(constraints.numNonLinearInequalityConstraints()),
  data_lower(mxl + mli + mni),
  data_upper(mxu)
{
}


auto ComplementarityVariables::wrtCanonicalLowerBounds() const -> VectorConstRef
{
    return data_lower;
}

auto ComplementarityVariables::wrtCanonicalLowerBounds() -> VectorRef
{
    return data_lower;
}


auto ComplementarityVariables::wrtCanonicalUpperBounds() const -> VectorConstRef
{
    return data_upper;
}

auto ComplementarityVariables::wrtCanonicalUpperBounds() -> VectorRef
{
    return data_upper;
}


auto ComplementarityVariables::wrtLowerBounds() const -> VectorConstRef
{
    return data_lower.head(mxl);
}

auto ComplementarityVariables::wrtLowerBounds() -> VectorRef
{
    return data_lower.head(mxl);
}


auto ComplementarityVariables::wrtUpperBounds() const -> VectorConstRef
{
    return data_upper.head(mxu);
}

auto ComplementarityVariables::wrtUpperBounds() -> VectorRef
{
    return data_upper.head(mxu);
}


auto ComplementarityVariables::wrtLinearInequalityConstraints() const -> VectorConstRef
{
    return data_lower.segment(mxl, mli);
}

auto ComplementarityVariables::wrtLinearInequalityConstraints() -> VectorRef
{
    return data_lower.segment(mxl, mli);
}


auto ComplementarityVariables::wrtNonLinearInequalityConstraints() const -> VectorConstRef
{
    return data_lower.tail(mni);
}

auto ComplementarityVariables::wrtNonLinearInequalityConstraints() -> VectorRef
{
    return data_lower.tail(mni);
}


} // namespace Optima
