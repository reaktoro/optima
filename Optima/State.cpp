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

#include "State.hpp"

// Optima includes
#include <Optima/IndexUtils.hpp>

namespace Optima {

struct State::Impl
{
    /// The variables \eq{\bar{x} = (x, u, v)} of the basic optimization problem.
    Vector xbar;

    /// The Lagrange multipliers \eq{y=(y_{b_{\mathrm{e}}},y_{b_{\mathrm{g}}},y_{h_{\mathrm{e}}},y_{h_{\mathrm{g}}})} of the optimization problem.
    Vector ybar;

    /// The variables \eq{\bar{z}=(z,y_{b_{\mathrm{g}}},y_{h_{\mathrm{g}}})} of the basic optimization problem.
    Vector zbar;

    /// Construct a State::Impl object with given dimensions information.
    Impl(const Dims& dims)
    : xbar(zeros(dims.x + dims.bg + dims.hg)),
      ybar(zeros(dims.be + dims.bg + dims.he + dims.hg)),
      zbar(zeros(dims.x + dims.bg + dims.hg))
    {}
};

State::State(const Dims& dims)
: pimpl(new Impl(dims)),
  dims(dims),
  x(pimpl->xbar.head(dims.x)),
  y(pimpl->ybar),
  ybe(pimpl->ybar.head(dims.be)),
  ybg(pimpl->ybar.segment(dims.be, dims.bg)),
  yhe(pimpl->ybar.segment(dims.be + dims.bg, dims.he)),
  yhg(pimpl->ybar.tail(dims.hg)),
  z(pimpl->zbar.head(dims.x)),
  xbar(pimpl->xbar),
  zbar(pimpl->zbar),
  xbg(pimpl->xbar.segment(dims.x, dims.bg)),
  xhg(pimpl->xbar.tail(dims.hg)),
  stability({indices(dims.x), dims.x})
{}

State::State(const State& other)
: pimpl(new Impl(*other.pimpl)),
  dims(other.dims),
  x(pimpl->xbar.head(other.dims.x)),
  y(pimpl->ybar),
  ybe(pimpl->ybar.head(other.dims.be)),
  ybg(pimpl->ybar.segment(other.dims.be, other.dims.bg)),
  yhe(pimpl->ybar.segment(other.dims.be + other.dims.bg, other.dims.he)),
  yhg(pimpl->ybar.tail(other.dims.hg)),
  z(pimpl->zbar.head(other.dims.x)),
  xbar(pimpl->xbar),
  zbar(pimpl->zbar),
  xbg(pimpl->xbar.segment(other.dims.x, other.dims.bg)),
  xhg(pimpl->xbar.tail(other.dims.hg)),
  stability(other.stability)
{}

State::~State()
{}

} // namespace Optima
