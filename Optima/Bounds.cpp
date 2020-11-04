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

#include "Bounds.hpp"

// Eigen includes
#include <Optima/deps/eigen3/Eigen/Dense>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct Bounds::Impl
{
    /// The lower bounds of the variables.
    Vector lower;

    /// The upper bounds of the variables.
    Vector upper;

    /// Construct a Bounds::Impl instance with given dimension.
    Impl(Index size)
    : lower(constants(size, -infinity())), upper(constants(size,  infinity()))
    {}

    /// Construct a Bounds::Impl instance with given lower and upper bounds.
    Impl(VectorConstRef lower, VectorConstRef upper)
    : lower(lower), upper(upper)
    {}

    /// Calculate the minimum distance to the lower or upper bounds.
    auto distance(VectorConstRef p) const -> Vector
    {
        Vector dist = min(p - lower, upper - p);

        dist = dist.array().isInf().select(abs(p), dist);

        return dist;
    }
};

Bounds::Bounds(Index size)
: pimpl(new Impl(size))
{}

Bounds::Bounds(VectorConstRef lower, VectorConstRef upper)
: pimpl(new Impl(lower, upper))
{
}

Bounds::Bounds(const Bounds& other)
: pimpl(new Impl(*other.pimpl))
{}

Bounds::~Bounds()
{}

auto Bounds::operator=(Bounds other) -> Bounds&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto Bounds::lower() const -> VectorConstRef
{
    return pimpl->lower;
}

auto Bounds::upper() const -> VectorConstRef
{
    return pimpl->upper;
}

} // namespace Optima
