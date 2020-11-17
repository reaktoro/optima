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

#include "ObjectiveResult.hpp"

namespace Optima {

struct ObjectiveResult::Impl
{
    /// The evaluated gradient of *f(x, p)* with respect to *x*.
    Vector fx;

    /// The evaluated Jacobian *fx(x, p)* with respect to *x*.
    Matrix fxx;

    /// The evaluated Jacobian *fx(x, p)* with respect to *p*.
    Matrix fxp;

    /// Construct a ObjectiveResult::Impl object.
    Impl(Index nx, Index np)
    {
        fx = zeros(nx);
        fxx = zeros(nx, nx);
        fxp = zeros(nx, np);
    }
};

ObjectiveResult::ObjectiveResult(Index nx, Index np)
: pimpl(new Impl(nx, np)),
  fx(pimpl->fx),
  fxx(pimpl->fxx),
  fxp(pimpl->fxp)
{}

ObjectiveResult::ObjectiveResult(const ObjectiveResult& other)
: pimpl(new Impl(*other.pimpl)),
  fx(pimpl->fx),
  fxx(pimpl->fxx),
  fxp(pimpl->fxp)
{}

ObjectiveResult::~ObjectiveResult()
{}

} // namespace Optima
