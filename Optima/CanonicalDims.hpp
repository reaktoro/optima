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

// Optima includes
#include <Optima/Index.hpp>

namespace Optima {

/// The dimension details of a canonical master matrix.
struct CanonicalDims
{
    const Index nx;  ///< The number of variables x.
    const Index np;  ///< The number of variables p.
    const Index ny;  ///< The number of variables y.
    const Index nz;  ///< The number of variables z.
    const Index nw;  ///< The number of variables w = (y, z).
    const Index nt;  ///< The number of variables (x, p, w).
    const Index ns;  ///< The number of stable variables in x.
    const Index nu;  ///< The number of unstable variables in x.
    const Index nb;  ///< The number of basic variables in x.
    const Index nn;  ///< The number of non-basic variables in x.
    const Index nl;  ///< The number of linearly dependent rows in Wx = [Ax; Jx].
    const Index nbs; ///< The number of stable basic variables.
    const Index nbu; ///< The number of unstable basic variables.
    const Index nns; ///< The number of stable non-basic variables.
    const Index nnu; ///< The number of unstable non-basic variables.
    const Index nbe; ///< The number of stable explicit basic variables.
    const Index nbi; ///< The number of stable implicit basic variables.
    const Index nne; ///< The number of stable explicit non-basic variables.
    const Index nni; ///< The number of stable implicit non-basic variables.
};

} // namespace Optima
