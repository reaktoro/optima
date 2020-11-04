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

#pragma once

// Optima includes
#include <Optima/Index.hpp>

namespace Optima {

/// Used to represent the dimensions in a master matrix.
struct MasterDims
{
    const Index nx; ///< The number of variables *x*.
    const Index np; ///< The number of variables *p*.
    const Index ny; ///< The number of variables *y*.
    const Index nz; ///< The number of variables *z*.
    const Index nw; ///< The number of variables *w = (y, z)*.
    const Index nt; ///< The total number of variables in *(x, p, y, z)*.

    /// Construct a MasterDims object with given dimensions.
    MasterDims(Index nx, Index np, Index ny, Index nz)
    : nx(nx), np(np), ny(ny), nz(nz), nw(ny + nz), nt(nx + np + nw) {}
};

} // namespace Optima
