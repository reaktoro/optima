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

#include "LagrangeMultipliers.hpp"

// Optima includes
#include <Optima/Constraints.hpp>

namespace Optima {

LagrangeMultipliers::LagrangeMultipliers()
{}

LagrangeMultipliers::LagrangeMultipliers(const Constraints& constraints)
: mle(constraints.numLinearEqualityConstraints()),
  mne(constraints.numNonLinearEqualityConstraints()),
  mli(constraints.numLinearInequalityConstraints()),
  mni(constraints.numNonLinearInequalityConstraints()),
  data(zeros(mle + mli + mne + mni))
{
}

auto LagrangeMultipliers::canonical() const -> VectorConstRef
{
    return data;
}

auto LagrangeMultipliers::canonical() -> VectorRef
{
    return data;
}


auto LagrangeMultipliers::wrtLinearEqualityConstraints() const -> VectorConstRef
{
    return data.head(mle);
}

auto LagrangeMultipliers::wrtLinearEqualityConstraints() -> VectorRef
{
    return data.head(mle);
}


auto LagrangeMultipliers::wrtNonLinearEqualityConstraints() const -> VectorConstRef
{
    return data.segment(mle, mne);
}

auto LagrangeMultipliers::wrtNonLinearEqualityConstraints() -> VectorRef
{
    return data.segment(mle, mne);
}


auto LagrangeMultipliers::wrtLinearInequalityConstraints() const -> VectorConstRef
{
    return data.segment(mle + mne, mli);
}

auto LagrangeMultipliers::wrtLinearInequalityConstraints() -> VectorRef
{
    return data.segment(mle + mne, mli);
}


auto LagrangeMultipliers::wrtNonLinearInequalityConstraints() const -> VectorConstRef
{
    return data.tail(mni);
}

auto LagrangeMultipliers::wrtNonLinearInequalityConstraints() -> VectorRef
{
    return data.tail(mni);
}

} // namespace Optima
