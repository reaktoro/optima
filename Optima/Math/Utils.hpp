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

// C++ includes
#include <tuple>

// Optima includes
#include <Optima/Math/Matrix.hpp>

namespace Optima {

/// Calculate the inverse of `A + D` where `inv(A)` is already known and `D` is a diagonal matrix.
/// @param invA[in,out] The inverse of the matrix `A` and the final inverse of `A + D`
/// @param D The diagonal matrix `D`
auto inverseShermanMorrison(const Matrix& invA, const Vector& D) -> Matrix;

/// Calculates the rational number that approximates a given real number.
/// The algorithm is based on Farey sequence as shown
/// [here](http://www.johndcook.com/blog/2010/10/20/best-rational-approximation/).
/// @param x The real number.
/// @param n The maximum denominator.
auto rationalize(double x, unsigned n) -> std::tuple<long, long>;

} // namespace Optima
