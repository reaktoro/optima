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
  xc(zeros(dims.x, nc)),
  pc(zeros(dims.p, nc)),
  yec(zeros(dims.be, nc)),
  ygc(zeros(dims.bg, nc)),
  zec(zeros(dims.he, nc)),
  zgc(zeros(dims.hg, nc)),
  sc(zeros(dims.x, nc)),
  xbgc(zeros(dims.bg, nc)),
  xhgc(zeros(dims.hg, nc))
{}

auto Sensitivity::operator=(const Sensitivity& other) -> Sensitivity&
{
    const_cast<Dims&>(dims) = other.dims;
    xc.__assign(other.xc);
    pc.__assign(other.pc);
    yec.__assign(other.yec);
    ygc.__assign(other.ygc);
    zec.__assign(other.zec);
    zgc.__assign(other.zgc);
    sc.__assign(other.sc);
    xbgc.__assign(other.xbgc);
    xhgc.__assign(other.xhgc);
    return *this;
}

} // namespace Optima
