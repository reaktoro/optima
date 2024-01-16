// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
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
#include <Optima/Utils.hpp>

namespace Optima {

Problem::Problem()
{}

Problem::Problem(const Dims& dims)
: dims(dims),
  Aex(zeros(dims.be, dims.x)),
  Aep(zeros(dims.be, dims.p)),
  Agx(zeros(dims.bg, dims.x)),
  Agp(zeros(dims.bg, dims.p)),
  be(zeros(dims.be)),
  bg(zeros(dims.bg)),
  xlower(constants(dims.x, -infinity())),
  xupper(constants(dims.x, infinity())),
  plower(constants(dims.p, -infinity())),
  pupper(constants(dims.p, infinity())),
  c(zeros(dims.c)),
  bec(zeros(dims.be, dims.c)),
  bgc(zeros(dims.bg, dims.c))
{}

auto Problem::operator=(const Problem& other) -> Problem&
{
    const_cast<Dims&>(dims) = other.dims;
    Aex.__assign(other.Aex);
    Aep.__assign(other.Aep);
    Agx.__assign(other.Agx);
    Agp.__assign(other.Agp);
    be.__assign(other.be);
    bg.__assign(other.bg);
    xlower.__assign(other.xlower);
    xupper.__assign(other.xupper);
    plower.__assign(other.plower);
    pupper.__assign(other.pupper);
    c.__assign(other.c);
    bec.__assign(other.bec);
    bgc.__assign(other.bgc);
    return *this;
}

} // namespace Optima
