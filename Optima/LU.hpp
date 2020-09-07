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

    /// Solve the linear system `AX = B` using the LU decomposition obtained with @ref decompose.
    /// @note Ensure method @ref decompose has been called before this method.
    auto solve(MatrixConstRef B, MatrixRef X) -> void;

    /// Solve the linear system `AX = B` using the LU decomposition obtained with @ref decompose.
    /// @param[in,out] X As input, matrix `B`. As output, matrix `X`.
    /// @note Ensure method @ref decompose has been called before this method.
    auto solve(MatrixRef X) -> void;

    /// Solve the linear system `Ax = b` with a scaling strategy for increased robustness.
    /// This method should be used when scaling of the LU decomposition of `A`
    /// needs to take into account vector `b` for scaling purposes. This allows
    /// determination of which equations are trully linearly dependent when
    /// solving `Ax = b`.
    /// @note This method does not require @ref decompose to be called beforehand.
    /// @warning Once this method is called, any previous @ref decompose call is no longer recorded.
    /// @warning Any small value in `b` is assumed here to be meaningful. This
    /// means that small values as a result of residual round off-error should
    /// have been cleaned off before this method is called. Otherwise, this
    /// small value may produce incorrect solutions and wrong behavior when
    /// identifying linearly dependent equations.
    auto solveWithScaling(MatrixConstRef A, VectorConstRef b, VectorRef x) -> void;

    /// Return the rank of the last LU decomposed matrix.
    /// @note Ensure method @ref decompose or @ref solveWithScaling has been called before this method.
    auto rank() const -> Index;

    /// Return the matrix containing the lower and upper triangular factors.
    auto matrixLU() const -> MatrixConstRef;

    /// Return the permutation matrix factor *P* of the LU decomposition *PAQ = LU*.
    auto P() const -> PermutationMatrix;

    /// Return the permutation matrix factor *P* of the LU decomposition *PAQ = LU*.
    auto Q() const -> PermutationMatrix;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
