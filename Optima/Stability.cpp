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
    auto is_meta_stable    = [&](Index i) { return is_lower_unstable(i) || is_upper_unstable(i); };

    //---------------------------------------------------------------------------------------------
    // NOTE
    //---------------------------------------------------------------------------------------------
    // In the code below, all basic variables are, by design, considered
    // stable. It is possible, however, that some basic variables are located
    // on their lower or upper bounds at the end of the calculation (in
    // degenerate cases). For means of computation, these are still classified
    // as stable (i.e., they are considered in the linear system of equations
    // to compute a Newton direction). However, they are also classified as
    // *meta-stable*. When we compute sensitivity derivatives for the optimum
    // state with respect to input variables, the derivatives associated with
    // the meta-stable variables are zero.
    //
    // The code below also identifies which non-basic variables are unstable
    // and if they lower or upper unstable.
    //---------------------------------------------------------------------------------------------

    // Initialize the vector of indices jsu with ordered indices from 0 to `nx - 1`
    jsu.noalias() = indices(nx);

    // Move basic variables to the left, leaving non-basic ones on the right
    // Initialize the number of basic stable variables (equivalent to number of basic variables)
    nbs = moveIntersectionLeft(jsu, jb);

    // Ensure there are no repeated basic indices in `jb`
    assert(nbs == jb.size());

    // Create reference to the first `nbs` entries in `jsu`, which corresponds to stable basic variables.
    auto jbs = jsu.head(nbs);

    // Move the meta-stable basic variables to the right in `jbs`, leaving the strictly stable basic variables on the left.
    const auto nss = moveRightIf(jbs, is_meta_stable);

    // Compute the number of non-basic variables (remember, all basic variables are stable, nb === nbs, but among some of these, there could be meta-stable basic variables).
    const auto nn = nx - nbs;

    // Create reference to the last `nn` entries in `jsu`, corresponding to the non-basic variables.
    auto jnsu = jsu.tail(nn);

    // Organize the non-basic variables in `jnsu` as (jns, jlu, juu) = (stable, lower unstable, upper unstable).
    const auto pos1 = moveRightIf(jnsu, is_upper_unstable);
    const auto pos2 = moveRightIf(jnsu.head(pos1), is_lower_unstable);

    // Initialize the number of stable variables (accounting for both stable basic and non-basic)
    ns  = nbs + pos2;

    // Initialize the number of upper unstable non-basic variables
    nuu = nn - pos1;

    // Initialize the number of lower unstable non-basic variables
    nlu = pos1 - pos2;

    // Initialize the number of meta-stable basic variables `nms`.
    nms = nbs - nss;

    // Ensure the number of stable, upper unstable, and lower unstable equals number of variables x
    assert(ns + nuu + nlu == nx);
}

auto Stability::status() const -> StabilityStatus
{
    const auto js = jsu.head(ns);
    const auto jbs = js.head(nbs);
    const auto jms = jbs.tail(nms);
    const auto ju = jsu.tail(nlu + nuu);
    const auto jlu = ju.head(nlu);
    const auto juu = ju.tail(nuu);
    return {js, ju, jlu, juu, jms, s};
}

} // namespace Optima
