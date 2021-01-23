// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2020 Allan Leal
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

#include "Sensitivity.hpp"

namespace Optima {

Sensitivity::Sensitivity()
{}

Sensitivity::Sensitivity(const Dims& dims, Index nc)
: dims(dims),
  dxdc(zeros(dims.x, nc)),
  dpdc(zeros(dims.p, nc)),
  dyedc(zeros(dims.be, nc)),
  dygdc(zeros(dims.bg, nc)),
  dzedc(zeros(dims.he, nc)),
  dzgdc(zeros(dims.hg, nc)),
  dsdc(zeros(dims.x, nc)),
  dxbgdc(zeros(dims.bg, nc)),
  dxhgdc(zeros(dims.hg, nc))
{}

auto Sensitivity::operator=(const Sensitivity& other) -> Sensitivity&
{
    const_cast<Dims&>(dims) = other.dims;
    dxdc.__assign(other.dxdc);
    dpdc.__assign(other.dpdc);
    dyedc.__assign(other.dyedc);
    dygdc.__assign(other.dygdc);
    dzedc.__assign(other.dzedc);
    dzgdc.__assign(other.dzgdc);
    dsdc.__assign(other.dsdc);
    dxbgdc.__assign(other.dxbgdc);
    dxhgdc.__assign(other.dxhgdc);
    return *this;
}

} // namespace Optima
