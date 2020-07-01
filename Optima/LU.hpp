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

/// A class for a more stable solution of linear systems using LU decomposition.
struct LU
{
    /// Construct a default LU object.
    LU();

    /// Construct an LU object with given matrix.
    explicit LU(MatrixConstRef A);

    /// Construct a copy of an LU object.
    LU(const LU& other);

    /// Destroy this LU object.
    virtual ~LU();

    /// Assign an LU object to this.
    auto operator=(LU other) -> LU&;

    /// Return true if empty.
    auto empty() const -> bool;

    /// Compute the LU decomposition of the given matrix.
    auto decompose(MatrixConstRef A) -> void;

    /// Solve the linear system `AX = B` using the calculated LU decomposition.
    auto solve(MatrixConstRef B, MatrixRef X) -> void;

    /// Solve the linear system `AX = B` using the calculated LU decomposition.
    auto solve(MatrixRef X) -> void;

    /// Return the rank of the matrix.
    /// @note: Method @ref decompose needs to be called first.
    auto rank() const -> Index;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
