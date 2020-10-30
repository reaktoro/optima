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

// C++ includes
#include <memory>

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to represent the canonical vector *us = (xs, p, wbs)*.
class CanonicalVector
{
public:
    VectorConstRef xs;  ///< The vector xs in the canonical vector.
    VectorConstRef p;   ///< The vector p in the canonical vector.
    VectorConstRef wbs; ///< The vector wbs in the canonical vector.
};

/// Used to represent the canonical vector *us = (xs, p, wbs)*.
class CanonicalVectorRef
{
public:
    VectorRef xs;  ///< The vector xs in the canonical vector.
    VectorRef p;   ///< The vector p in the canonical vector.
    VectorRef wbs; ///< The vector wbs in the canonical vector.
};

/// Used to represent the vector *u = (x, p, y, z)*.
class MasterVector
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a MasterVector instance.
    MasterVector(Index nx, Index np, Index ny, Index nz);

    /// Construct a copy of a MasterVector instance.
    MasterVector(const MasterVector& other);

    /// Destroy this MasterVector instance.
    virtual ~MasterVector();

    /// Assign a MasterVector instance to this.
    auto operator=(MasterVector other) -> MasterVector&;

    VectorRef x; ///< The reference to the vector segment *x* in *u = (x, p, y, z)*.
    VectorRef p; ///< The reference to the vector segment *p* in *u = (x, p, y, z)*.
    VectorRef y; ///< The reference to the vector segment *y* in *u = (x, p, y, z)*.
    VectorRef z; ///< The reference to the vector segment *z* in *u = (x, p, y, z)*.
    VectorRef w; ///< The reference to the vector segment *w = (y, z)* in *u = (x, p, y, z)*.

    VectorRef data; ///< The access to the underlying vector data in *u = (x, p, y, z)*.
};

} // namespace Optima
