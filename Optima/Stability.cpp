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

#include "Stability.hpp"

// C++ includes
#include <cassert>

// Optima includes
#include <Optima/IndexUtils.hpp>

namespace Optima {

Stability::Stability()
{}

Stability::Stability(Index nx)
: jsu(indices(nx)), ns(nx), nlu(0), nuu(0), s(zeros(nx))
{}

auto Stability::update(StabilityUpdateArgs args) -> void
{
    const auto [Wx, g, x, w, xlower, xupper, jb] = args;

    const auto nx = x.size();

    assert(nx == g.size());
    assert(nx == xlower.size());
    assert(nx == xupper.size());

    s.noalias() = g + tr(Wx)*w;

    auto is_lower_unstable = [&](Index i) { return args.x[i] == args.xlower[i] && s[i] > 0.0; };
    auto is_upper_unstable = [&](Index i) { return args.x[i] == args.xupper[i] && s[i] < 0.0; };

    // Note: In the code below, all basic variables are by default considered
    // stable. It remains to identify which non-basic variables are unstable!

    jsu.noalias() = indices(nx);

    const auto nb = moveIntersectionLeft(jsu, jb);

    assert(nb == jb.size());

    const auto nn = nx - nb; // the number of non-basic variables

    auto jnsu = jsu.tail(nn); // the non-basic variables organized as jnsu = (jns, jnu) = (stable, unstable)

    // Organize jsu = (js, jlu, juu) = (stable, lower unstable, upper unstable).
    const auto pos1 = moveRightIf(jnsu, is_upper_unstable);
    const auto pos2 = moveRightIf(jnsu.head(pos1), is_lower_unstable);

    ns  = nb + pos2;
    nuu = nn - pos1;
    nlu = pos1 - pos2;

    assert(ns + nuu + nlu == nx);
}

auto Stability::status() const -> StabilityStatus
{
    const auto js = jsu.head(ns);
    const auto ju = jsu.tail(nlu + nuu);
    const auto jlu = ju.head(nlu);
    const auto juu = ju.tail(nuu);
    return {js, ju, jlu, juu, s};
}

} // namespace Optima
