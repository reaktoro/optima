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

/// A type used to describe a matrix @f$ A @f$ in canonical form.
/// The canonical form of a matrix @f$ A @f$ is represented as:
/// @f[
/// C = RA=\begin{bmatrix}I_{\mathrm{b}} & C_{\mathrm{n}}\end{bmatrix}Q,
/// @f]
/// where @f$ Q @f$ is a permutation matrix, and @f$ R @f$ is the
/// *canonicalizer matrix* of @f$ A @f$.
struct CanonicalMatrix
{
    /// The matrix @f$ C_{\mathrm{n}} @f$.
    Matrix Cn;

    /// The permutation matrix @f$ Q @f$.
    PermutationMatrix Q;

    /// The canonicalizer matrix @f$ R @f$.
    Matrix R;

    /// The inverse of the canonicalizer matrix @f$ R @f$.
    Matrix invR;
};

auto canonicalize(const Matrix& A, CanonicalMatrix& C) -> void;

} // namespace Optima
