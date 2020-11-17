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

#include "ConstraintResult.hpp"

namespace Optima {

struct ConstraintResult::Impl
{
    /// The evalauted vector value of *c(x, p)*.
    Vector val;

    /// The evalauted Jacobian matrix of *c(x, p)* with respect to *x*.
    Matrix ddx;

    /// The evalauted Jacobian matrix of *c(x, p)* with respect to *p*.
    Matrix ddp;

    /// Construct a ConstraintResult::Impl object.
    Impl(Index nc, Index nx, Index np)
    {
        val = zeros(nc);
        ddx = zeros(nc, nx);
        ddp = zeros(nc, np);
    }
};

ConstraintResult::ConstraintResult(Index nc, Index nx, Index np)
: pimpl(new Impl(nc, nx, np)),
  val(pimpl->val),
  ddx(pimpl->ddx),
  ddp(pimpl->ddp)
{}

ConstraintResult::ConstraintResult(const ConstraintResult& other)
: pimpl(new Impl(*other.pimpl)),
  val(pimpl->val),
  ddx(pimpl->ddx),
  ddp(pimpl->ddp)
{}

ConstraintResult::~ConstraintResult()
{}

} // namespace Optima
