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

/// Used to represent the dimensions in a master matrix.
struct MasterDims
{
    const Index nx; ///< The number of primal variables *x*.
    const Index np; ///< The number of unknonw parameter variables *p*.
    const Index ny; ///< The number of Lagrange multiplier variables *y*.
    const Index nz; ///< The number of Lagrange multiplier variables *z*.
    const Index nw; ///< The number of Lagrange multiplier variables *w = (y, z)*.
    const Index nt; ///< The total number of unknown variables in *u = (x, p, y, z)*.

    /// Construct a default MasterDims object.
    MasterDims()
    : MasterDims(0, 0, 0, 0) {}

    /// Construct a MasterDims object with given dimensions.
    MasterDims(Index nx, Index np, Index ny, Index nz)
    : nx(nx), np(np), ny(ny), nz(nz), nw(ny + nz), nt(nx + np + nw) {}

    /// Assign another MasterDims object to this.
    auto operator=(const MasterDims& other) -> MasterDims&
    {
        const_cast<Index&>(nx) = other.nx;
        const_cast<Index&>(np) = other.np;
        const_cast<Index&>(ny) = other.ny;
        const_cast<Index&>(nz) = other.nz;
        const_cast<Index&>(nw) = other.nw;
        const_cast<Index&>(nt) = other.nt;
        return *this;
    }
};

} // namespace Optima
