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

#pragma once

namespace Optima {

/// The options for the backtrack search operations.
struct BacktrackSearchOptions
{
    /// The flag that indicates if a simpler approach for fixing out-of-bounds
    /// variables should be used. Setting this option to true will cause the
    /// updated variables `x` and `p` after a Newton step to be fixed with `x =
    /// min(xlower, max(x, xupper))` and `p = min(plower, max(p, pupper))` and
    /// immediately accepted. If line-search is to be performed afterwards, it
    /// is possible that the new direction vector from previous to updated `u`
    /// vector is not a descent direction, since the min-max fix alters its
    /// orientation.
    bool apply_min_max_fix_and_accept = false;
};

} // namespace Optima
