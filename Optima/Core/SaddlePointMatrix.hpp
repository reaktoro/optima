// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2016 Allan Leal
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
};

/// A type used to describe a saddle point right-hand side vector.
struct SaddlePointVector
{
    /// The right-hand side vector `a`.
    Vector a;

    /// The right-hand side vector `b`.
    Vector b;

    /// The right-hand side vector `c`.
    Vector c;
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
};

/// A type used to describe a canonical saddle point right-hand side vector.
struct SaddlePointVectorCanonical
{
    /// The right-hand side vector `a = [ab, as, au]` of the canonical problem.
    Vector ab, as, au;

    /// The right-hand side vector `b` of the canonical problem.
    Vector b;

    /// The right-hand side vector `c = [cb, cs, cu]` of the canonical problem.
    Vector cb, cs, cu;
};

} // namespace Optima
