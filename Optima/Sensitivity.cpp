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

#include "Sensitivity.hpp"

namespace Optima {

Sensitivity::Sensitivity()
{}

Sensitivity::Sensitivity(const Dims& dims)
: dims(dims),
  xc(zeros(dims.x, dims.c)),
  pc(zeros(dims.p, dims.c)),
  yec(zeros(dims.be, dims.c)),
  ygc(zeros(dims.bg, dims.c)),
  zec(zeros(dims.he, dims.c)),
  zgc(zeros(dims.hg, dims.c)),
  sc(zeros(dims.x, dims.c)),
  xbgc(zeros(dims.bg, dims.c)),
  xhgc(zeros(dims.hg, dims.c))
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

auto Sensitivity::resize(const Dims& newdims) -> void
{
    const_cast<Dims&>(dims) = newdims;
    xc.__resize(dims.x, dims.c);
    pc.__resize(dims.p, dims.c);
    yec.__resize(dims.be, dims.c);
    ygc.__resize(dims.bg, dims.c);
    zec.__resize(dims.he, dims.c);
    zgc.__resize(dims.hg, dims.c);
    sc.__resize(dims.x, dims.c);
    xbgc.__resize(dims.bg, dims.c);
    xhgc.__resize(dims.hg, dims.c);
}

} // namespace Optima
