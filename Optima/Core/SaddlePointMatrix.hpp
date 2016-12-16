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
    Vector G;

    /// The diagonal matrix `Bb` in the canonical coefficient matrix.
    Vector Bb;

    /// The matrix `Bn` in `B = [Bb Bn]` in the canonical coefficient matrix.
    Matrix Bn;

    /// The diagonal matrix `E = diag(Eb, Es, Eu)` in the coefficient matrix.
    Vector E;

    /// The indices of the variables describing their ordering in the canonical form.
    Indices ordering;

    /// The number of basic variables.
    Index nb;

    /// The number of non-basic variables.
    Index nn;

    /// The number of stable non-basic variables.
    Index ns;

    /// The number of unstable non-basic variables.
    Index nu;

    /// Convert this SaddlePointMatrixCanonical instance into a Matrix instance.
    auto convert() const -> Matrix;

    /// Return `true` if this SaddlePointMatrixCanonical instance is valid.
    auto valid() const -> bool;
};

} // namespace Optima

//
//he coefficient matrix of a saddle point problem.
//class SaddlePointMatrix
//{
//public:
//    /// The Hessian matrix \eq{H} in the saddle point matrix.
//    HessianMatrix H;
//
//    /// The Jacobian matrix \eq{A} in the saddle point matrix.
//    Matrix A;
//
//    /// The diagonal matrix \eq{X} in the saddle point matrix.
//    Vector X;
//
//    /// The diagonal matrix \eq{Z} in the saddle point matrix.
//    Vector Z;
//
//public:
//    /// Construct a default SaddlePointMatrix instance.
//    SaddlePointMatrix();
//
//    /// Convert this SaddlePointMatrix instance into a Matrix instance.
//    auto convert() const -> Matrix;
//
//    /// Return `true` if this SaddlePointMatrix instance is valid.
//    auto valid() const -> bool;
//
//private:
//    /// The scaled Hessian matrix `G` partitioned in order of basic, stable, and unstable variables.
//    HessianMatrix G;
//
//    /// The scaled diagonal matrix \eq{E = -XZ} partitioned in order of basic, stable, and unstable variables.
//    Vector E;
//
//    /// The diagonal matrix `Bb` in `B = [Bb Bs Bu]`, where `B = RAQX`, with `R` and `Q` obtained from canonicalization.
//    Vector Bb;
//
//    /// The matrix `Bn = [Bs Bu]` in `B = [Bb Bs Bu]`, where `B = RAQX`, with `R` and `Q` obtained from canonicalization.
//    Matrix Bn;
//
//    /// The canonicalizer of the Jacobian matrix `A` used to compute `B` and to determine basic and non-basic variables.
//    Canonicalizer canonicalizer;
//};

