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

#include "Stability.hpp"

// C++ includes
#include <cassert>

namespace Optima {

/// Check if the data of a Stability object is consistent.
auto checkStabilityConsistency(const Stability::Data& data)
{
    const auto& [iordering, ns, nlu, nuu, nslu, nsuu] = data;
    const auto size = iordering.size();
    assert(ns <= size);
    assert(nlu <= size);
    assert(nuu <= size);
    assert(nslu <= size);
    assert(nsuu <= size);
    assert(size == ns + nlu + nuu + nslu + nsuu);
}

Stability::Stability()
{}

Stability::Stability(const Data& data)
: data(data)
{
    checkStabilityConsistency(data);
}

auto Stability::update(const Data& data_) -> void
{
    data = data_;
    checkStabilityConsistency(data);
}

auto Stability::numVariables() const -> Index
{
    return data.iordering.size();
}

auto Stability::numStableVariables() const -> Index
{
    return data.ns;
}

auto Stability::numUnstableVariables() const -> Index
{
    return data.iordering.size() - data.ns;
}

auto Stability::numLowerUnstableVariables() const -> Index
{
    return data.nlu;
}

auto Stability::numUpperUnstableVariables() const -> Index
{
    return data.nuu;
}

auto Stability::numStrictlyLowerUnstableVariables() const -> Index
{
    return data.nslu;
}

auto Stability::numStrictlyUpperUnstableVariables() const -> Index
{
    return data.nsuu;
}

auto Stability::numStrictlyUnstableVariables() const -> Index
{
    return data.nslu + data.nsuu;
}

auto Stability::indicesVariables() const -> IndicesView
{
    return data.iordering;
}

auto Stability::indicesStableVariables() const -> IndicesView
{
    return data.iordering.head(data.ns);
}

auto Stability::indicesUnstableVariables() const -> IndicesView
{
    return data.iordering.tail(data.nlu + data.nuu + data.nslu + data.nsuu);
}

auto Stability::indicesLowerUnstableVariables() const -> IndicesView
{
    return data.iordering.segment(data.ns, data.nlu);
}

auto Stability::indicesUpperUnstableVariables() const -> IndicesView
{
    return data.iordering.segment(data.ns + data.nlu, data.nuu);
}

auto Stability::indicesStrictlyLowerUnstableVariables() const -> IndicesView
{
    return data.iordering.segment(data.ns + data.nlu + data.nuu, data.nslu);
}

auto Stability::indicesStrictlyUpperUnstableVariables() const -> IndicesView
{
    return data.iordering.tail(data.nsuu);
}

auto Stability::indicesStrictlyUnstableVariables() const -> IndicesView
{
    return data.iordering.tail(data.nslu + data.nsuu);
}

} // namespace Optima
