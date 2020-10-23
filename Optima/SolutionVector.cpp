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

#include "SolutionVector.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/JacobianMatrix.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct SolutionVector::Impl
{
    const Index nx = 0;   ///< The number of variables x.
    const Index np = 0;   ///< The number of variables p.
    const Index ny = 0;   ///< The number of variables y.
    const Index nz = 0;   ///< The number of variables z.
    const Index nw = 0;   ///< The number of variables w = (y, z).
    Vector u;             ///< The vector u = (x, p, y, z).

    Impl(Index nx, Index np, Index ny, Index nz)
    : nx(nx), np(np), ny(ny), nz(nz), nw(ny + nz),
      u(zeros(nx + np + ny + nz))
    {
    }

    auto x() -> VectorRef { return u.head(nx); }
    auto p() -> VectorRef { return u.segment(nx, np); }
    auto y() -> VectorRef { return u.segment(nx + np, ny); }
    auto z() -> VectorRef { return u.segment(nx + np + ny, nz); }
    auto w() -> VectorRef { return u.tail(ny + nz); }
};

SolutionVector::SolutionVector(Index nx, Index np, Index ny, Index nz)
: pimpl(new Impl(nx, np, ny, nz)),
  x(pimpl->x()),
  p(pimpl->p()),
  y(pimpl->y()),
  z(pimpl->z()),
  w(pimpl->w()),
  vec(pimpl->u)
{}

SolutionVector::SolutionVector(const SolutionVector& other)
: pimpl(new Impl(*other.pimpl)),
  x(pimpl->x()),
  p(pimpl->p()),
  y(pimpl->y()),
  z(pimpl->z()),
  w(pimpl->w()),
  vec(pimpl->u)
{}

SolutionVector::~SolutionVector()
{}

auto SolutionVector::operator=(SolutionVector other) -> SolutionVector&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

} // namespace Optima
