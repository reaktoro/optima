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

#include "State.hpp"

namespace Optima {

State::State()
{}

State::State(const Dims& dims)
: dims(dims),
  x(zeros(dims.x)),
  p(zeros(dims.p)),
  ye(zeros(dims.be)),
  yg(zeros(dims.bg)),
  ze(zeros(dims.he)),
  zg(zeros(dims.hg)),
  s(zeros(dims.x)),
  xbg(zeros(dims.bg)),
  xhg(zeros(dims.hg))
{}

auto State::operator=(const State& other) -> State&
{
    const_cast<Dims&>(dims) = other.dims;
    x.__assign(other.x);
    p.__assign(other.p);
    ye.__assign(other.ye);
    yg.__assign(other.yg);
    ze.__assign(other.ze);
    zg.__assign(other.zg);
    s.__assign(other.s);
    xbg.__assign(other.xbg);
    xhg.__assign(other.xhg);
    js = other.js;
    ju = other.ju;
    jlu = other.jlu;
    juu = other.juu;
    jb = other.jb;
    jn = other.jn;

    return *this;
}

} // namespace Optima
