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

/// Used to represent the solution vector *u = (x, p, y, z)* in the optimization problem.
class SolutionVector
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a SolutionVector instance.
    SolutionVector(Index nx, Index np, Index ny, Index nz);

    /// Construct a copy of a SolutionVector instance.
    SolutionVector(const SolutionVector& other);

    /// Destroy this SolutionVector instance.
    virtual ~SolutionVector();

    /// Assign a SolutionVector instance to this.
    auto operator=(SolutionVector other) -> SolutionVector&;

    VectorRef x; ///< The entries in the solution vector corresponding to *x* variables.
    VectorRef p; ///< The entries in the solution vector corresponding to *p* variables.
    VectorRef y; ///< The entries in the solution vector corresponding to *y* variables.
    VectorRef z; ///< The entries in the solution vector corresponding to *z* variables.
    VectorRef w; ///< The entries in the solution vector corresponding to *w = (y, z)* variables.

    VectorRef vec; ///< The access to the underlying vector data in *u = (x, p, y, z)*.
};

} // namespace Optima
