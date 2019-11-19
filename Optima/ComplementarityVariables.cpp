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
: n(constraints.numVariables()),
  mli(constraints.numLinearInequalityConstraints()),
  mni(constraints.numNonLinearInequalityConstraints()),
  data(zeros(n + mli + mni))
{
}


auto ComplementarityVariables::canonical() const -> VectorConstRef
{
    return data;
}

auto ComplementarityVariables::canonical() -> VectorRef
{
    return data;
}


auto ComplementarityVariables::original() const -> VectorConstRef
{
    return data.head(n);
}

auto ComplementarityVariables::original() -> VectorRef
{
    return data.head(n);
}


auto ComplementarityVariables::wrtLinearInequalityConstraints() const -> VectorConstRef
{
    return data.segment(n, mli);
}

auto ComplementarityVariables::wrtLinearInequalityConstraints() -> VectorRef
{
    return data.segment(n, mli);
}


auto ComplementarityVariables::wrtNonLinearInequalityConstraints() const -> VectorConstRef
{
    return data.tail(mni);
}

auto ComplementarityVariables::wrtNonLinearInequalityConstraints() -> VectorRef
{
    return data.tail(mni);
}


} // namespace Optima
