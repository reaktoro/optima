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

#include "StablePartition.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>

namespace Optima {

struct StablePartition::Impl
{
    /// The indices of the variables ordered as jsu = (js, ju)
    Indices jsu;

    /// The number of stable variables in js.
    Index ns;

    /// Construct a StablePartition::Impl instance with given dimension.
    Impl(Index nx)
    : jsu(indices(nx)), ns(nx)
    {}

    auto setStable(IndicesView js) -> void
    {
        ns = moveIntersectionLeft(jsu, js);
        assert(ns == js.size() && "There are repeated or out-of-bound indices in js");
    }

    auto setUnstable(IndicesView ju) -> void
    {
        ns = moveIntersectionRight(jsu, ju);
        assert(ns == jsu.size() - ju.size() && "There are repeated or out-of-bound indices in ju");
    }

    auto stable() const -> IndicesView
    {
        return jsu.head(ns);
    }

    auto unstable() const -> IndicesView
    {
        const auto nu = jsu.size() - ns;
        return jsu.tail(nu);
    }
};

StablePartition::StablePartition(Index size)
: pimpl(new Impl(size))
{}

StablePartition::StablePartition(const StablePartition& other)
: pimpl(new Impl(*other.pimpl))
{}

StablePartition::~StablePartition()
{}

auto StablePartition::operator=(StablePartition other) -> StablePartition&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto StablePartition::setStable(IndicesView js) -> void
{
    pimpl->setStable(js);
}

auto StablePartition::setUnstable(IndicesView ju) -> void
{
    pimpl->setUnstable(ju);
}

auto StablePartition::stable() const -> IndicesView
{
    return pimpl->stable();
}

auto StablePartition::unstable() const -> IndicesView
{
    return pimpl->unstable();
}

} // namespace Optima
