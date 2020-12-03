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

#include "State.hpp"

// Optima includes
#include <Optima/IndexUtils.hpp>

namespace Optima {

struct State::Impl
{
    /// The variables x(bar) = (x, u, v) of the basic optimization problem.
    Vector xbar;

    /// The parameter variables p of the basic optimization problem.
    Vector p;

    /// The Lagrange multipliers w = (y, z) = (ye, yg, ze, zg) of the optimization problem.
    Vector wbar;

    /// The variables \eq{(s,y_{\mathrm{g}},z_{\mathrm{g}})} of the basic optimization problem.
    Vector sbar;

    /// Construct a State::Impl object with given dimensions information.
    Impl(const Dims& dims)
    : xbar(zeros(dims.x + dims.bg + dims.hg)),
      p(zeros(dims.p)),
      wbar(zeros(dims.be + dims.bg + dims.he + dims.hg)),
      sbar(zeros(dims.x + dims.bg + dims.hg))
    {}
};

State::State(const Dims& dims)
: pimpl(new Impl(dims)),
  dims(dims),
  x(pimpl->xbar.head(dims.x)),
  p(pimpl->p),
  w(pimpl->wbar),
  y(pimpl->wbar.head(dims.be + dims.bg)),
  ye(y.head(dims.be)),
  yg(y.tail(dims.bg)),
  z(pimpl->wbar.tail(dims.he + dims.hg)),
  ze(z.head(dims.he)),
  zg(z.tail(dims.hg)),
  s(pimpl->sbar.head(dims.x)),
  xbar(pimpl->xbar),
  sbar(pimpl->sbar),
  xbg(pimpl->xbar.segment(dims.x, dims.bg)),
  xhg(pimpl->xbar.tail(dims.hg)),
  stability(dims.x)
{}

State::State(const State& other)
: pimpl(new Impl(*other.pimpl)),
  dims(other.dims),
  x(pimpl->xbar.head(other.dims.x)),
  p(pimpl->p),
  w(pimpl->wbar),
  y(pimpl->wbar.head(dims.be + dims.bg)),
  ye(y.head(dims.be)),
  yg(y.tail(dims.bg)),
  z(pimpl->wbar.tail(dims.he + dims.hg)),
  ze(z.head(dims.he)),
  zg(z.tail(dims.hg)),
  s(pimpl->sbar.head(other.dims.x)),
  xbar(pimpl->xbar),
  sbar(pimpl->sbar),
  xbg(pimpl->xbar.segment(other.dims.x, other.dims.bg)),
  xhg(pimpl->xbar.tail(other.dims.hg)),
  stability(other.stability)
{}

State::~State()
{}

} // namespace Optima
