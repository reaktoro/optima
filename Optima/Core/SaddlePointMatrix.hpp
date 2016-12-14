// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2017 Allan Leal
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
#include <Optima/Math/Matrix.hpp>

namespace Optima {

/// A type used to describe a saddle point coefficient matrix.
struct SaddlePointMatrix
{
    /// The diagonal matrix `H` in the coefficient matrix.
    Vector H;

    /// The matrix `A` in the coefficient matrix.
    Matrix A;

    /// The diagonal matrix `X` in the coefficient matrix.
    Vector X;

    /// The diagonal matrix `Z` in the coefficient matrix.
    Vector Z;

    /// Convert this SaddlePointMatrix instance into a Matrix instance.
    auto convert() const -> Matrix;

    /// Return `true` if this SaddlePointMatrix instance is valid.
    auto valid() const -> bool;
};

/// A type used to describe a saddle point right-hand side vector.
struct SaddlePointVector
{
    /// The saddle-point vector `x`.
    Vector x;

    /// The saddle-point vector `y`.
    Vector y;

    /// The saddle-point vector `z`.
    Vector z;

    /// Convert this SaddlePointVector instance into a Vector instance.
    auto convert() const -> Vector;

    /// Return `true` if this SaddlePointMatrix instance is valid.
    auto valid() const -> bool;
};

/// A type used to describe a canonical saddle point coefficient matrix.
struct SaddlePointMatrixCanonical
{
    /// The diagonal matrix `G = diag(Gb, Gs, Gu)` in the coefficient matrix.
    Vector Gb, Gs, Gu;

    /// The diagonal matrix `Bb` in the canonical coefficient matrix.
    Vector Bb;

    /// The matrix `B = [Bb Bs Bu]` in the canonical coefficient matrix.
    Matrix Bs, Bu;

    /// The diagonal matrix `E = diag(Eb, Es, Eu)` in the coefficient matrix.
    Vector Eb, Es, Eu;

    /// Convert this SaddlePointMatrixCanonical instance into a Matrix instance.
    auto convert() const -> Matrix;

    /// Return `true` if this SaddlePointMatrixCanonical instance is valid.
    auto valid() const -> bool;
};

/// A type used to describe a canonical saddle point right-hand side vector.
struct SaddlePointVectorCanonical
{
    /// The canonical saddle point vector `x = [xb, xs, xu]`.
    Vector xb, xs, xu;

    /// The canonical saddle point vector `y`.
    Vector y;

    /// The canonical saddle point vector `z = [zb, zs, zu]`.
    Vector zb, zs, zu;

    /// Convert this SaddlePointVectorCanonical instance into a Vector instance.
    auto convert() const -> Vector;

    /// Return `true` if this SaddlePointVectorCanonical instance is valid.
    auto valid() const -> bool;
};

} // namespace Optima
